# core/evaluator.py
import math
import chess
from collections import defaultdict
from engine.config import PIECE_VALUES

# --- Piece-square tables (midgame / endgame) in centipawns (example small tables) ---
# For brevity I keep small tables; you should tune these to play style or import from a file.
KNIGHT_PST_MG = [
 -50,-40,-30,-30,-30,-30,-40,-50,
 -40,-20,  0,  5,  5,  0,-20,-40,
 -30,  5, 10, 15, 15, 10,  5,-30,
 -30,  0, 15, 20, 20, 15,  0,-30,
 -30,  5, 15, 20, 20, 15,  5,-30,
 -30,  0, 10, 15, 15, 10,  0,-30,
 -40,-20,  0,  0,  0,  0,-20,-40,
 -50,-40,-30,-30,-30,-30,-40,-50
]
KNIGHT_PST_EG = [ -30 ]*64  # simpler in endgame

PAWN_PST_MG = [0]*64
PAWN_PST_EG = [0]*64

KING_PST_MG = [0]*64
KING_PST_EG = [0]*64

# tuning constants
BISHOP_PAIR_BONUS = 50
OUTPOST_KNIGHT_BONUS = 40
ROOK_OPEN_FILE_BONUS = 20
ROOK_HALF_OPEN_FILE_BONUS = 10
HANGING_PENALTY = 80
BACKWARD_PAWN_PENALTY = 30

# phase weights for piece types (how much they contribute to game phase)
PHASE_WEIGHTS = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

class Evaluator:
    def __init__(self, use_positional=True, use_tactics=True, use_king_safety=True):
        self.use_positional = use_positional
        self.use_tactics = use_tactics
        self.use_king_safety = use_king_safety

    # -------------------------
    # Main entry
    # -------------------------
    def evaluate_board(self, board: chess.Board) -> int:
        """
        Return evaluation in centipawns (White positive).
        This function is intentionally not incremental — it computes a full eval.
        """
        # piece map once
        piece_map = board.piece_map()
        
        pieces_by_type_color = { (pt, color): board.pieces(pt, color) for pt in range(1,7) for color in [chess.WHITE, chess.BLACK] }
        
        white_attacks = [board.attackers(chess.WHITE, sq) for sq in range(64)]
        black_attacks = [board.attackers(chess.BLACK, sq) for sq in range(64)]


        # compute game phase (0 = endgame, 1 = opening/mid)
        phase = self._game_phase(piece_map, board)

        material = self._material(piece_map)
        positional = self._positional(piece_map, phase) if self.use_positional else 0
        mobility = self._mobility(board, piece_map)
        threats = self._threats(board, piece_map)
        pawns = self._pawn_structure(board, piece_map)
        king = self._king_safety(board, piece_map, phase) if self.use_king_safety else 0
        tactics = self._tactics(board, pieces_by_type_color, white_attacks, black_attacks) if self.use_tactics else 0

        total = material + positional + mobility + threats + pawns + king + tactics

        # small normalization to avoid noisy floats
        return int(total)

    # -------------------------
    # Game phase calculation (0..1)
    # -------------------------
    def _game_phase(self, piece_map, board: chess.Board):
        """
        Compute game phase as a float 0..1 (1 = opening/midgame, 0 = endgame)
        """
        max_phase = sum(v * 1 for v in PHASE_WEIGHTS.values())
        phase_sum = 0
        for sq, p in piece_map.items():
            phase_sum += PHASE_WEIGHTS.get(p.piece_type, 0)
            
        if max_phase == 0:
            return 0.0

        pg = phase_sum / 54 + board.fullmove_number / 16  # 48 chosen to map typical midgame near 1; tune if needed
        return max(0.0, min(1.0, pg))

    # -------------------------
    # Material (centipawns) - simple
    # -------------------------
    def _material(self, piece_map):
        score = 0
        for sq, p in piece_map.items():
            v = PIECE_VALUES.get(p.piece_type, 0)
            score += v if p.color == chess.WHITE else -v
        # bishop pair bonus
        for color in (chess.WHITE, chess.BLACK):
            bishops = len(list(filter(lambda x: piece_map[x].piece_type == chess.BISHOP and piece_map[x].color == color, piece_map)))
            if bishops >= 2:
                score += BISHOP_PAIR_BONUS if color == chess.WHITE else -BISHOP_PAIR_BONUS
        return score

    # -------------------------
    # Positional (PST + activity)
    # -------------------------
    def _positional(self, piece_map, phase):
        score = 0
        # piece-square tables use midgame vs endgame mix
        for sq, p in piece_map.items():
            if p.piece_type == chess.KNIGHT:
                val_mg = KNIGHT_PST_MG[sq]
                val_eg = KNIGHT_PST_EG[sq]
                val = int(val_mg * phase + val_eg * (1 - phase))
            elif p.piece_type == chess.PAWN:
                val_mg = PAWN_PST_MG[sq]; val_eg = PAWN_PST_EG[sq]
                val = int(val_mg * phase + val_eg * (1 - phase))
            elif p.piece_type == chess.KING:
                val_mg = KING_PST_MG[sq]; val_eg = KING_PST_EG[sq]
                val = int(val_mg * phase + val_eg * (1 - phase))
            else:
                val = 0
            score += val if p.color == chess.WHITE else -val

        # outpost knights: a knight on a square where enemy pawns do not control but supported by own pawn
        for color in (chess.WHITE, chess.BLACK):
            enemy = not color
            knight_pos = list(filter(lambda s: piece_map[s].piece_type == chess.KNIGHT 
                                     and piece_map[s].color == color, piece_map))
            for sq in knight_pos:
                f = chess.square_file(sq); r = chess.square_rank(sq)
                # outpost heuristic: no enemy pawn attacks this square, and supported by own pawn (adjacent behind)
                enemy_pawn_attack = False
                for df in (-1,0,1):
                    of = f + df
                    rr = r + (1 if color == chess.WHITE else -1)
                    if 0 <= of <= 7 and 0 <= rr <= 7:
                        if chess.square(of, rr) in (list(filter(lambda s: piece_map[s].piece_type == chess.PAWN and piece_map[s].color == enemy, piece_map))):
                            enemy_pawn_attack = True; break
                if not enemy_pawn_attack:
                    # supported by own pawn?
                    supported = False
                    for df in (-1,1):
                        of = f + df; rr = r - (1 if color == chess.WHITE else -1)
                        if 0 <= of <= 7 and 0 <= rr <= 7:
                            if chess.square(of, rr) in (list(filter(lambda s: piece_map[s].piece_type == chess.PAWN and piece_map[s].color == color, piece_map))):
                                supported = True; break
                    if supported:
                        score += OUTPOST_KNIGHT_BONUS if color == chess.WHITE else -OUTPOST_KNIGHT_BONUS

        # rooks on open / half-open files
        # precompute file occupancy
        file_occupancy = [0]*8
        for s in piece_map:
            file_occupancy[chess.square_file(s)] += 1
        for sq, p in piece_map.items():
            if p.piece_type == chess.ROOK:
                f = chess.square_file(sq)
                if file_occupancy[f] == 1:  # only the rook on that file => open file (ignores opponent king)
                    score += ROOK_OPEN_FILE_BONUS if p.color == chess.WHITE else -ROOK_OPEN_FILE_BONUS
                elif file_occupancy[f] <= 2:
                    score += ROOK_HALF_OPEN_FILE_BONUS if p.color == chess.WHITE else -ROOK_HALF_OPEN_FILE_BONUS

        return score

    # -------------------------
    # Mobility (light-weight; count pseudo-legal moves per piece)
    # -------------------------
    def _mobility(self, board, piece_map):
        # weight per-piece (in centipawns per legal move)
        weights = {
            chess.PAWN: 3,
            chess.KNIGHT: 15,
            chess.BISHOP: 12,
            chess.ROOK: 8,
            chess.QUEEN: 6,
            chess.KING: 2
        }
        counts = defaultdict(int)
        # iterate pseudo-legal moves once
        for m in board.pseudo_legal_moves:
            counts[m.from_square] += 1
        score = 0
        for sq, p in piece_map.items():
            cnt = counts.get(sq, 0)
            w = weights.get(p.piece_type, 0)
            score += cnt * w if p.color == chess.WHITE else -cnt * w
        return score

    # -------------------------
    # Threats (attackers/defenders) - scaled modestly
    # -------------------------
    def _threats(self, board, piece_map):
        score = 0
        atk_weight = {
            chess.PAWN: 25,
            chess.KNIGHT: 60,
            chess.BISHOP: 60,
            chess.ROOK: 90,
            chess.QUEEN: 150,
            chess.KING: 0
        }
        def_weight = {
            chess.PAWN: 10,
            chess.KNIGHT: 25,
            chess.BISHOP: 25,
            chess.ROOK: 40,
            chess.QUEEN: 80,
            chess.KING: 0
        }
        # Use board.attackers(color, sq) directly
        for sq, p in piece_map.items():
            attackers = board.attackers(not p.color, sq)
            defenders = board.attackers(p.color, sq)
            # sum weighted attacker values
            atk_score = 0
            for a in attackers:
                pa = board.piece_at(a)
                if pa:
                    atk_score += atk_weight.get(pa.piece_type, 20)
            def_score = 0
            for d in defenders:
                pd = board.piece_at(d)
                if pd:
                    def_score += def_weight.get(pd.piece_type, 10)
            if p.color == chess.WHITE:
                score -= atk_score
                score += def_score
            else:
                score += atk_score
                score -= def_score
        return score

    # -------------------------
    # Pawn structure
    # -------------------------
    def _pawn_structure(self, board, piece_map):
        score = 0
        # files and ranks per color
        files_w = defaultdict(int)
        files_b = defaultdict(int)
        pawns_w = set()
        pawns_b = set()
        for sq, p in piece_map.items():
            if p.piece_type == chess.PAWN:
                if p.color == chess.WHITE:
                    files_w[chess.square_file(sq)] += 1
                    pawns_w.add(sq)
                else:
                    files_b[chess.square_file(sq)] += 1
                    pawns_b.add(sq)

        # doubled penalty
        for f, c in files_w.items():
            if c > 1:
                score -= 30 * (c - 1)
        for f, c in files_b.items():
            if c > 1:
                score += 30 * (c - 1)

        # isolated pawn penalty
        for f in range(8):
            if files_w.get(f,0) > 0 and files_w.get(f-1,0)==0 and files_w.get(f+1,0)==0:
                score -= 25 * files_w.get(f,0)
            if files_b.get(f,0) > 0 and files_b.get(f-1,0)==0 and files_b.get(f+1,0)==0:
                score += 25 * files_b.get(f,0)

        # backward pawns (a pawn that cannot be safely advanced because of enemy pawn control)
        # simple heuristic: pawn with no friendly pawn behind it and square ahead attacked by enemy pawn
        for sq in pawns_w:
            f = chess.square_file(sq); r = chess.square_rank(sq)
            behind = False
            for rr in range(0, r):
                if chess.square(f, rr) in pawns_w:
                    behind = True; break
            if not behind:
                # check if forward-square is attacked by enemy pawn
                forward = None
                if r+1 <= 7:
                    forward = chess.square(f, r+1)
                if forward is not None:
                    # if enemy pawn attacks forward
                    attacked_by_enemy_pawn = False
                    for df in (-1,1):
                        of = f + df; rr = r+1
                        if 0 <= of <=7 and 0 <= rr <=7:
                            if chess.square(of, rr) in pawns_b:
                                attacked_by_enemy_pawn = True; break
                    if attacked_by_enemy_pawn:
                        score -= BACKWARD_PAWN_PENALTY

        for sq in pawns_b:
            f = chess.square_file(sq); r = chess.square_rank(sq)
            behind = False
            for rr in range(r+1,8):
                if chess.square(f, rr) in pawns_b:
                    behind = True; break
            if not behind:
                forward = None
                if r-1 >= 0:
                    forward = chess.square(f, r-1)
                if forward is not None:
                    attacked_by_enemy_pawn = False
                    for df in (-1,1):
                        of = f + df; rr = r-1
                        if 0 <= of <=7 and 0 <= rr <=7:
                            if chess.square(of, rr) in pawns_w:
                                attacked_by_enemy_pawn = True; break
                    if attacked_by_enemy_pawn:
                        score += BACKWARD_PAWN_PENALTY

        # passed pawn detection (cleaner using pawn sets)
        for sq in pawns_w:
            f = chess.square_file(sq); r = chess.square_rank(sq)
            blocked = False
            for of in range(max(0,f-1), min(7,f+1)+1):
                for rr in range(r+1, 8):
                    tsq = chess.square(of, rr)
                    if tsq in pawns_b:
                        blocked = True; break
                if blocked: break
            if not blocked:
                score += 60  # bigger passed pawn bonus; tune
                # connected passed pawn bonus: check adjacent friendly pawns ahead
                for of in (f-1, f+1):
                    if 0 <= of <= 7:
                        for rr in range(r+1,8):
                            tsq = chess.square(of, rr)
                            if tsq in pawns_w:
                                score += 20; break

        for sq in pawns_b:
            f = chess.square_file(sq); r = chess.square_rank(sq)
            blocked = False
            for of in range(max(0,f-1), min(7,f+1)+1):
                for rr in range(r-1, -1, -1):
                    tsq = chess.square(of, rr)
                    if tsq in pawns_w:
                        blocked = True; break
                if blocked: break
            if not blocked:
                score -= 60
                for of in (f-1, f+1):
                    if 0 <= of <= 7:
                        for rr in range(r-1,-1,-1):
                            tsq = chess.square(of, rr)
                            if tsq in pawns_b:
                                score -= 20; break

        return score

    # -------------------------
    # King safety
    # -------------------------
    def _king_safety(self, board, piece_map, phase):
        """
        King safety uses a king-zone heuristic plus pawn shelter and heavy pieces aiming at king files.
        Phase mixes: in endgame king activity is good; in midgame safety is critical.
        """
        score = 0
        for color in (chess.WHITE, chess.BLACK):
            ksq = board.king(color)
            if ksq is None:
                continue
            kf = chess.square_file(ksq); kr = chess.square_rank(ksq)
            zone = []
            for df in (-1,0,1):
                for dr in (-1,0,1):
                    nf, nr = kf+df, kr+dr
                    if 0 <= nf <= 7 and 0 <= nr <= 7:
                        zone.append(chess.square(nf,nr))
            attacker_value = 0
            defender_value = 0
            heavy_attacks = 0
            for z in zone:
                attackers = board.attackers(not color, z)
                defenders = board.attackers(color, z)
                for a in attackers:
                    p = board.piece_at(a)
                    if p:
                        v = {chess.PAWN:25, chess.KNIGHT:50, chess.BISHOP:50, chess.ROOK:80, chess.QUEEN:140}.get(p.piece_type,30)
                        attacker_value += v
                        if p.piece_type in (chess.QUEEN, chess.ROOK):
                            heavy_attacks += 1
                for d in defenders:
                    p = board.piece_at(d)
                    if p:
                        defender_value += {chess.PAWN:10, chess.KNIGHT:30, chess.BISHOP:30, chess.ROOK:50, chess.QUEEN:90}.get(p.piece_type,10)
            # pawn shelter: count pawns directly in front of king (3 squares)
            shield = 0
            if color == chess.WHITE:
                for df in (-1,0,1):
                    nf = kf + df; nr = kr + 1
                    if 0 <= nf <= 7 and 0 <= nr <= 7 and board.piece_at(chess.square(nf,nr)) and board.piece_at(chess.square(nf,nr)).piece_type == chess.PAWN and board.piece_at(chess.square(nf,nr)).color == color:
                        shield += 1
            else:
                for df in (-1,0,1):
                    nf = kf + df; nr = kr - 1
                    if 0 <= nf <= 7 and 0 <= nr <= 7 and board.piece_at(chess.square(nf,nr)) and board.piece_at(chess.square(nf,nr)).piece_type == chess.PAWN and board.piece_at(chess.square(nf,nr)).color == color:
                        shield += 1
            # open file / semi-open file towards the king (heavy pieces on file)
            file_threat = 0
            # scan vertically (same file)
            kf = chess.square_file(ksq)
            kr = chess.square_rank(ksq)
            for r in range(8):
                if r == kr:
                    continue
                sq = chess.square(kf, r)
                p = board.piece_at(sq)
                if p and p.color != color and p.piece_type in (chess.ROOK, chess.QUEEN):
                    file_threat += 1

            # scan horizontally (same rank)
            for f in range(8):
                if f == kf:
                    continue
                sq = chess.square(f, kr)
                p = board.piece_at(sq)
                if p and p.color != color and p.piece_type in (chess.ROOK, chess.QUEEN):
                    file_threat += 1


            ks = defender_value - int(attacker_value * max(0.2, 1.0 - 0.2*shield))
            # weights: in opening/mid (phase ~1) king safety matters more, in endgame king activity is good
            weight = 1.0 * phase + -0.5 * (1 - phase)  # midgame -> +1, endgame -> -0.5 (king activity rewarded)
            if color == chess.WHITE:
                score += int(ks * weight) - file_threat*40
            else:
                score -= int(ks * weight) - file_threat*40
            # penalty for many heavy attacks near the king
            if heavy_attacks >= 2:
                score += (-120 if color == chess.WHITE else 120)
        return score

    # -------------------------
    # Simple tactical checks: hanging pieces, simple capture-sequence heuristics
    # -------------------------
    def _tactics(self, board, piece_by_type_color, white_attacks, black_attacks):
        score = 0
        for (pt, color), squares in piece_by_type_color.items():
            for sq in squares:
                attackers = white_attacks[sq] if color == chess.WHITE else black_attacks[sq]
                if not attackers:
                    continue

                defenders = board.attackers(color, sq)

                attacked_gain = sum(
                    PIECE_VALUES.get(board.piece_at(a).piece_type, 0)
                    for a in attackers if board.piece_at(a)
                )
                defended_gain = sum(
                    PIECE_VALUES.get(board.piece_at(d).piece_type, 0)
                    for d in defenders if board.piece_at(d)
                )

                SEE = attacked_gain - defended_gain - PIECE_VALUES.get(pt, 0)

                if SEE > 0:
                    score += int(SEE * 0.1) if color == chess.WHITE else -int(SEE * 0.1)
                else:
                    score -= int(abs(SEE) * 0.5) if color == chess.WHITE else -int(abs(SEE) * 0.5)

        # simple capture heuristics
        for mv in board.generate_legal_captures():
            victim = board.piece_at(mv.to_square)
            attacker = board.piece_at(mv.from_square)
            if not victim or not attacker:
                continue
            gain = PIECE_VALUES.get(victim.piece_type, 0) - PIECE_VALUES.get(attacker.piece_type, 0)
            if gain > 0:
                score += int(gain * 0.6) if attacker.color == chess.WHITE else -int(gain * 0.6)
            elif gain < 0:
                score += int(gain * 0.2) if attacker.color == chess.WHITE else -int(gain * 0.2)

        return score
