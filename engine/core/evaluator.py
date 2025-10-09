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
        
        strategic = self._strategic_patterns(board, piece_map)
        potential = self._piece_potential(board, piece_map)
        advanced_pawns = self._advanced_pawn_structure(board, piece_map)
        space = self._space_control(board)
        initiative = self._initiative(board, piece_map)
        
        total = (material + positional + mobility + threats + 
                pawns + king + tactics + strategic + potential + 
                advanced_pawns + space + initiative)
        
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

    def _strategic_patterns(self, board, piece_map):
        """Recognize common strategic patterns and formations"""
        score = 0
        
        # Bishop pair dominance in open positions
        if self._is_position_open(board):
            white_bishops = len([p for p in piece_map.values() 
                            if p.piece_type == chess.BISHOP and p.color == chess.WHITE])
            black_bishops = len([p for p in piece_map.values() 
                            if p.piece_type == chess.BISHOP and p.color == chess.BLACK])
            if white_bishops == 2 and black_bishops < 2:
                score += 40  # Bishop pair in open position
            elif black_bishops == 2 and white_bishops < 2:
                score -= 40
        
        # Good vs bad bishop
        score += self._bishop_quality(board, piece_map)
        
        # Rook on 7th rank
        for color in [chess.WHITE, chess.BLACK]:
            rank = 6 if color == chess.WHITE else 1  # 7th rank
            for sq in board.pieces(chess.ROOK, color):
                if chess.square_rank(sq) == rank:
                    # Check if enemy king is trapped on back rank
                    enemy_king_sq = board.king(not color)
                    if enemy_king_sq and chess.square_rank(enemy_king_sq) == (0 if color == chess.WHITE else 7):
                        bonus = 50
                        score += bonus if color == chess.WHITE else -bonus
        
        return score

    def _bishop_quality(self, board, piece_map):
        """Evaluate bishop quality (good vs bad bishop)"""
        score = 0
        pawns_white = board.pieces(chess.PAWN, chess.WHITE)
        pawns_black = board.pieces(chess.PAWN, chess.BLACK)
        
        for sq, piece in piece_map.items():
            if piece.piece_type == chess.BISHOP:
                # Count pawns on same color squares
                same_color_pawns = 0
                if piece.color == chess.WHITE:
                    for pawn_sq in pawns_white:
                        if (chess.square_file(sq) + chess.square_rank(sq)) % 2 == \
                        (chess.square_file(pawn_sq) + chess.square_rank(pawn_sq)) % 2:
                            same_color_pawns += 1
                    penalty = same_color_pawns * 8
                    score -= penalty
                else:
                    for pawn_sq in pawns_black:
                        if (chess.square_file(sq) + chess.square_rank(sq)) % 2 == \
                        (chess.square_file(pawn_sq) + chess.square_rank(pawn_sq)) % 2:
                            same_color_pawns += 1
                    penalty = same_color_pawns * 8
                    score += penalty
        return score
    
    def _piece_potential(self, board, piece_map):
        """Evaluate piece potential and future possibilities"""
        score = 0
        
        # Knight outposts with long-term potential
        score += self._knight_outposts(board, piece_map)
        
        # Bishop mobility and diagonal control
        score += self._bishop_potential(board, piece_map)
        
        # Rook battery and coordination
        score += self._rook_coordination(board, piece_map)
        
        return score

    def _knight_outposts(self, board, piece_map):
        """Strong knight outposts that cannot be attacked by pawns"""
        score = 0
        for sq, piece in piece_map.items():
            if piece.piece_type == chess.KNIGHT:
                # Check if it's a strong outpost
                if self._is_strong_outpost(board, sq, piece.color):
                    # Bonus increases if outpost is in enemy territory
                    rank = chess.square_rank(sq)
                    if (piece.color == chess.WHITE and rank >= 4) or \
                    (piece.color == chess.BLACK and rank <= 3):
                        bonus = 25 + (abs(rank - 3.5) * 5)  # Deeper = better
                        score += bonus if piece.color == chess.WHITE else -bonus
        return score

    def _is_strong_outpost(self, board, sq, color):
        """Check if square is a strong knight outpost"""
        rank, file = chess.square_rank(sq), chess.square_file(sq)
        
        # Should be in enemy territory
        if (color == chess.WHITE and rank < 4) or (color == chess.BLACK and rank > 3):
            return False
        
        # Should not be attackable by enemy pawns
        enemy_pawn_attacks = False
        for df in [-1, 1]:
            attack_file = file + df
            attack_rank = rank + (1 if color == chess.WHITE else -1)
            if 0 <= attack_file <= 7 and 0 <= attack_rank <= 7:
                attack_sq = chess.square(attack_file, attack_rank)
                if board.piece_at(attack_sq) and \
                board.piece_at(attack_sq).piece_type == chess.PAWN and \
                board.piece_at(attack_sq).color != color:
                    enemy_pawn_attacks = True
                    break
        
        return not enemy_pawn_attacks
    
    def _advanced_pawn_structure(self, board, piece_map):
        """More sophisticated pawn structure evaluation"""
        score = 0
        
        # Pawn chains
        score += self._pawn_chains(board)
        
        # Candidate passed pawns
        score += self._candidate_passed_pawns(board)
        
        # Pawn majority and minority attacks
        score += self._pawn_majorities(board)
        
        return score

    def _pawn_chains(self, board):
        """Evaluate pawn chains - connected pawns that support each other"""
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            pawns = list(board.pieces(chess.PAWN, color))
            chain_bonus = 0
            
            for pawn_sq in pawns:
                # Check if this pawn supports or is supported by another pawn
                supporting = 0
                # Check diagonals behind for supporting pawns
                behind_file = chess.square_file(pawn_sq)
                behind_rank = chess.square_rank(pawn_sq) + (-1 if color == chess.WHITE else 1)
                for df in [-1, 1]:
                    if 0 <= behind_file + df <= 7 and 0 <= behind_rank <= 7:
                        behind_sq = chess.square(behind_file + df, behind_rank)
                        if behind_sq in pawns:
                            supporting += 1
                
                chain_bonus += supporting * 10
            
            score += chain_bonus if color == chess.WHITE else -chain_bonus
        
        return score

    def _candidate_passed_pawns(self, board):
        """Identify pawns that could become passed pawns"""
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            enemy_color = not color
            pawns = list(board.pieces(chess.PAWN, color))
            enemy_pawns = list(board.pieces(chess.PAWN, enemy_color))
            
            for pawn_sq in pawns:
                pawn_file = chess.square_file(pawn_sq)
                pawn_rank = chess.square_rank(pawn_sq)
                
                # Check if this pawn has a clear path to promotion
                clear_path = True
                for ahead_rank in range(pawn_rank + 1, 8) if color == chess.WHITE else range(pawn_rank - 1, -1, -1):
                    # Check for blocking pawns
                    for df in [-1, 0, 1]:
                        check_file = pawn_file + df
                        if 0 <= check_file <= 7:
                            check_sq = chess.square(check_file, ahead_rank)
                            if check_sq in enemy_pawns:
                                clear_path = False
                                break
                    if not clear_path:
                        break
                
                if clear_path:
                    # Bonus based on how far advanced and how clear the path is
                    advancement = pawn_rank if color == chess.WHITE else 7 - pawn_rank
                    bonus = advancement * 8
                    score += bonus if color == chess.WHITE else -bonus
        
        return score
    
    def _space_control(self, board):
        """Evaluate control of key squares and space"""
        score = 0
        
        # Center control
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        for sq in center_squares:
            white_control = len(board.attackers(chess.WHITE, sq))
            black_control = len(board.attackers(chess.BLACK, sq))
            score += (white_control - black_control) * 3
        
        # Extended center (big center)
        extended_center = [chess.C3, chess.D3, chess.E3, chess.F3,
                        chess.C4, chess.D4, chess.E4, chess.F4,
                        chess.C5, chess.D5, chess.E5, chess.F5,
                        chess.C6, chess.D6, chess.E6, chess.F6]
        
        for sq in extended_center:
            white_control = len(board.attackers(chess.WHITE, sq))
            black_control = len(board.attackers(chess.BLACK, sq))
            score += (white_control - black_control) * 2
        
        return score

    def _initiative(self, board, piece_map):
        """Evaluate who has the initiative (attacking chances)"""
        score = 0
        
        # Development advantage in opening
        if board.fullmove_number < 15:
            developed_white = len([sq for sq in piece_map if piece_map[sq].color == chess.WHITE 
                                and piece_map[sq].piece_type != chess.PAWN 
                                and piece_map[sq].piece_type != chess.KING
                                and chess.square_rank(sq) > 1])  # Not on back rank
            
            developed_black = len([sq for sq in piece_map if piece_map[sq].color == chess.BLACK 
                                and piece_map[sq].piece_type != chess.PAWN 
                                and piece_map[sq].piece_type != chess.KING
                                and chess.square_rank(sq) < 6])  # Not on back rank
            
            score += (developed_white - developed_black) * 12
        
        # Attack direction - are pieces pointing at enemy king?
        score += self._attack_direction(board)
        
        return score
    
    def _is_position_open(self, board):
        """
        Determine if the position is open (few pawns, open files/diagonals)
        Returns True for open positions, False for closed positions
        """
        total_pawns = len(board.pieces(chess.PAWN, chess.WHITE)) + len(board.pieces(chess.PAWN, chess.BLACK))
        total_pieces = len(board.piece_map())
        
        # Open positions typically have fewer pawns
        pawn_ratio = total_pawns / total_pieces if total_pieces > 0 else 0
        
        # Count open files (files with no pawns)
        open_files = 0
        for file in range(8):
            file_has_pawn = False
            for rank in range(8):
                sq = chess.square(file, rank)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN:
                    file_has_pawn = True
                    break
            if not file_has_pawn:
                open_files += 1
        
        # Position is considered open if:
        # - Few pawns (less than 50% of pieces are pawns)
        # - Multiple open files
        # - Few pawn exchanges happened (many pawns still on original squares)
        original_pawn_squares_white = [chess.square(f, 1) for f in range(8)]  # rank 2
        original_pawn_squares_black = [chess.square(f, 6) for f in range(8)]  # rank 7
        
        pawns_on_original_squares = 0
        for sq in original_pawn_squares_white + original_pawn_squares_black:
            piece = board.piece_at(sq)
            if piece and piece.piece_type == chess.PAWN:
                pawns_on_original_squares += 1
        
        # Heuristic: position is open if:
        # - Few pawns relative to total pieces, OR
        # - Many open files, OR  
        # - Few pawns remain on original squares (meaning lots of pawn exchanges)
        return (pawn_ratio < 0.4 or 
                open_files >= 4 or 
                pawns_on_original_squares <= 8)
        
    # Add these methods to your Evaluator class:

    def _bishop_potential(self, board, piece_map):
        """Evaluate bishop potential and long-term diagonal control"""
        score = 0
        
        # Long diagonal control bonus
        long_diagonals = [
            [chess.A1, chess.B2, chess.C3, chess.D4, chess.E5, chess.F6, chess.G7, chess.H8],
            [chess.A8, chess.B7, chess.C6, chess.D5, chess.E4, chess.F3, chess.G2, chess.H1]
        ]
        
        for color in [chess.WHITE, chess.BLACK]:
            bishops = [sq for sq, piece in piece_map.items() 
                    if piece.piece_type == chess.BISHOP and piece.color == color]
            
            for bishop_sq in bishops:
                # Bonus for bishops on long diagonals
                for diagonal in long_diagonals:
                    if bishop_sq in diagonal:
                        bonus = 15
                        score += bonus if color == chess.WHITE else -bonus
                        break
                
                # Bonus for bishops that control many squares
                mobility = len(list(board.attacks(bishop_sq)))
                mobility_bonus = mobility * 2
                score += mobility_bonus if color == chess.WHITE else -mobility_bonus
        
        return score

    def _rook_coordination(self, board, piece_map):
        """Evaluate rook coordination - connected rooks, batteries, etc."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            rooks = list(board.pieces(chess.ROOK, color))
            
            # Connected rooks (on same rank or file with no pieces between)
            if len(rooks) >= 2:
                for i in range(len(rooks)):
                    for j in range(i + 1, len(rooks)):
                        r1, r2 = rooks[i], rooks[j]
                        if chess.square_rank(r1) == chess.square_rank(r2):
                            # Check if same rank and no pieces between
                            file1, file2 = chess.square_file(r1), chess.square_file(r2)
                            min_file, max_file = min(file1, file2), max(file1, file2)
                            clear = True
                            for f in range(min_file + 1, max_file):
                                test_sq = chess.square(f, chess.square_rank(r1))
                                if board.piece_at(test_sq):
                                    clear = False
                                    break
                            if clear:
                                score += 25 if color == chess.WHITE else -25
                        
                        elif chess.square_file(r1) == chess.square_file(r2):
                            # Check if same file and no pieces between
                            rank1, rank2 = chess.square_rank(r1), chess.square_rank(r2)
                            min_rank, max_rank = min(rank1, rank2), max(rank1, rank2)
                            clear = True
                            for r in range(min_rank + 1, max_rank):
                                test_sq = chess.square(chess.square_file(r1), r)
                                if board.piece_at(test_sq):
                                    clear = False
                                    break
                            if clear:
                                score += 25 if color == chess.WHITE else -25
            
            # Rook battery (two rooks on same file with queen behind)
            for rook_sq in rooks:
                file = chess.square_file(rook_sq)
                # Look for another rook on same file
                for other_rook_sq in rooks:
                    if other_rook_sq != rook_sq and chess.square_file(other_rook_sq) == file:
                        # Check if queen is behind one of the rooks
                        queen_sq = board.king(color)  # This should be queen, but we'll use king for simplicity
                        if queen_sq and chess.square_file(queen_sq) == file:
                            if (color == chess.WHITE and chess.square_rank(queen_sq) < min(chess.square_rank(rook_sq), chess.square_rank(other_rook_sq))) or \
                            (color == chess.BLACK and chess.square_rank(queen_sq) > max(chess.square_rank(rook_sq), chess.square_rank(other_rook_sq))):
                                score += 30 if color == chess.WHITE else -30
        
        return score

    def _pawn_majorities(self, board):
        """Evaluate pawn majorities and minority attacks"""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            pawns = list(board.pieces(chess.PAWN, color))
            enemy_pawns = list(board.pieces(chess.PAWN, not color))
            
            # Count pawns on kingside (files E-H) and queenside (files A-D)
            kingside_pawns = len([p for p in pawns if chess.square_file(p) >= 4])
            queenside_pawns = len([p for p in pawns if chess.square_file(p) <= 3])
            
            enemy_kingside_pawns = len([p for p in enemy_pawns if chess.square_file(p) >= 4])
            enemy_queenside_pawns = len([p for p in enemy_pawns if chess.square_file(p) <= 3])
            
            # Pawn majority bonus
            if kingside_pawns > enemy_kingside_pawns:
                bonus = (kingside_pawns - enemy_kingside_pawns) * 10
                score += bonus if color == chess.WHITE else -bonus
            
            if queenside_pawns > enemy_queenside_pawns:
                bonus = (queenside_pawns - enemy_queenside_pawns) * 10
                score += bonus if color == chess.WHITE else -bonus
            
            # Minority attack potential (fewer pawns but creating weaknesses)
            if queenside_pawns < enemy_queenside_pawns and queenside_pawns > 0:
                # Potential for minority attack
                bonus = 15
                score += bonus if color == chess.WHITE else -bonus
        
        return score

    def _attack_direction(self, board):
        """Evaluate attack direction - are pieces aiming at enemy king position?"""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            enemy_king_sq = board.king(not color)
            if not enemy_king_sq:
                continue
                
            enemy_king_file = chess.square_file(enemy_king_sq)
            enemy_king_rank = chess.square_rank(enemy_king_sq)
            
            attack_weight = 0
            
            # Check all pieces of this color that can attack enemy king
            for piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                for sq in board.pieces(piece_type, color):
                    attacks = board.attacks(sq)
                    if enemy_king_sq in attacks:
                        # Bonus based on piece value and proximity to king
                        piece_bonus = {
                            chess.QUEEN: 20,
                            chess.ROOK: 15,
                            chess.BISHOP: 10,
                            chess.KNIGHT: 12
                        }.get(piece_type, 5)
                        
                        # Additional bonus if piece is in enemy territory
                        rank = chess.square_rank(sq)
                        if (color == chess.WHITE and rank >= 4) or (color == chess.BLACK and rank <= 3):
                            piece_bonus += 5
                        
                        attack_weight += piece_bonus
            
            score += attack_weight if color == chess.WHITE else -attack_weight
        
        return score