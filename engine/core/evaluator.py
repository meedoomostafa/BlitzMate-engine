# core/evaluator.py
import chess
from collections import defaultdict
from engine.config import PIECE_VALUES

# small positional PST examples (centipawns)
PAWN_PST = [0]*64
KNIGHT_PST = [ -50,-40,-30,-30,-30,-30,-40,-50,
               -40,-20,  0,  5,  5,  0,-20,-40,
               -30,  5, 10, 15, 15, 10,  5,-30,
               -30,  0, 15, 20, 20, 15,  0,-30,
               -30,  5, 15, 20, 20, 15,  5,-30,
               -30,  0, 10, 15, 15, 10,  0,-30,
               -40,-20,  0,  0,  0,  0,-20,-40,
               -50,-40,-30,-30,-30,-30,-40,-50 ]
BISHOP_PST = [0]*64
ROOK_PST = [0]*64
QUEEN_PST = [0]*64
KING_PST = [0]*64

class Evaluator:
    def __init__(self, use_positional=True):
        self.use_positional = use_positional

    def evaluate_board(self, board: chess.Board) -> int:
        """
        Return evaluation in centipawns (White positive).
        """
        piece_map = board.piece_map()  # cache once
        material = self._material(piece_map)
        positional = self._positional(piece_map) if self.use_positional else 0
        mobility = self._mobility(board, piece_map)
        threats = self._threats(board, piece_map)
        pawns = self._pawn_structure(board, piece_map)
        king = self._king_safety(board, piece_map)

        total = material + positional + mobility + threats + pawns + king
        return int(total)

    # -------------------------
    # Material (centipawns)
    # -------------------------
    def _material(self, piece_map):
        score = 0
        # count each piece
        counts_w = defaultdict(int)
        counts_b = defaultdict(int)
        for sq, p in piece_map.items():
            if p.color == chess.WHITE:
                counts_w[p.piece_type] += 1
            else:
                counts_b[p.piece_type] += 1
        for pt, val in PIECE_VALUES.items():
            score += (counts_w.get(pt,0) - counts_b.get(pt,0)) * val
        return score

    # -------------------------
    # Positional PST (small)
    # -------------------------
    def _positional(self, piece_map):
        score = 0
        for sq, p in piece_map.items():
            if p.piece_type == chess.KNIGHT:
                val = KNIGHT_PST[sq]
            elif p.piece_type == chess.PAWN:
                val = PAWN_PST[sq]
            else:
                val = 0
            score += val if p.color == chess.WHITE else -val
        return score

    # -------------------------
    # Mobility (light-weight)
    # -------------------------
    def _mobility(self, board, piece_map):
        # weight per-piece (in centipawns per legal move)
        weights = {
            chess.PAWN: 5,
            chess.KNIGHT: 15,
            chess.BISHOP: 15,
            chess.ROOK: 8,
            chess.QUEEN: 10,
            chess.KING: 3
        }
        counts = defaultdict(int)
        # use pseudo_legal_moves for speed (doesn't change board)
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
        # small attacker weight per piece type
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
        for sq, p in piece_map.items():
            attackers = list(board.attackers(not p.color, sq))
            defenders = list(board.attackers(p.color, sq))
            atk_score = sum(atk_weight.get(board.piece_at(a).piece_type, 20) for a in attackers if board.piece_at(a))
            def_score = sum(def_weight.get(board.piece_at(d).piece_type, 10) for d in defenders if board.piece_at(d))
            if p.color == chess.WHITE:
                score -= atk_score
                score += def_score
            else:
                score += atk_score
                score -= def_score
        return score

    # -------------------------
    # Pawn structure: doubled, isolated, passed (modest penalties/bonuses)
    # -------------------------
    def _pawn_structure(self, board, piece_map):
        score = 0
        files_w = defaultdict(int)
        files_b = defaultdict(int)
        for sq, p in piece_map.items():
            if p.piece_type == chess.PAWN:
                f = chess.square_file(sq)
                if p.color == chess.WHITE:
                    files_w[f] += 1
                else:
                    files_b[f] += 1
        # doubled penalty small
        for f, c in files_w.items():
            if c > 1:
                score -= 30 * (c - 1)
        for f, c in files_b.items():
            if c > 1:
                score += 30 * (c - 1)
        # isolated pawn penalty
        for f in range(8):
            if files_w.get(f,0) > 0 and files_w.get(f-1,0)==0 and files_w.get(f+1,0)==0:
                score -= 20 * files_w.get(f,0)
            if files_b.get(f,0) > 0 and files_b.get(f-1,0)==0 and files_b.get(f+1,0)==0:
                score += 20 * files_b.get(f,0)
        # passed pawns modest bonus
        for sq, p in piece_map.items():
            if p.piece_type != chess.PAWN: continue
            f = chess.square_file(sq); r = chess.square_rank(sq)
            if p.color == chess.WHITE:
                blocked = False
                for of in range(max(0,f-1), min(7,f+1)+1):
                    for rr in range(r+1,8):
                        tsq = chess.square(of, rr)
                        if tsq in board.pieces(chess.PAWN, chess.BLACK):
                            blocked = True; break
                    if blocked: break
                if not blocked:
                    score += 50
            else:
                blocked = False
                for of in range(max(0,f-1), min(7,f+1)+1):
                    for rr in range(r-1,-1,-1):
                        tsq = chess.square(of, rr)
                        if tsq in board.pieces(chess.PAWN, chess.WHITE):
                            blocked = True; break
                    if blocked: break
                if not blocked:
                    score -= 50
        return score

    # -------------------------
    # King safety (scaled)
    # -------------------------
    def _king_safety(self, board, piece_map):
        score = 0
        for color in [chess.WHITE, chess.BLACK]:
            ksq = board.king(color)
            if ksq is None:
                continue
            # define king zone
            kf = chess.square_file(ksq); kr = chess.square_rank(ksq)
            zone = []
            for df in (-1,0,1):
                for dr in (-1,0,1):
                    nf, nr = kf+df, kr+dr
                    if 0 <= nf <= 7 and 0 <= nr <= 7:
                        zone.append(chess.square(nf,nr))
            attacker_value = 0
            defender_value = 0
            for z in zone:
                attackers = board.attackers(not color, z)
                defenders = board.attackers(color, z)
                for a in attackers:
                    p = board.piece_at(a)
                    if p:
                        attacker_value += {chess.PAWN:25, chess.KNIGHT:50, chess.BISHOP:50, chess.ROOK:80, chess.QUEEN:140}.get(p.piece_type,30)
                for d in defenders:
                    p = board.piece_at(d)
                    if p:
                        defender_value += {chess.PAWN:10, chess.KNIGHT:30, chess.BISHOP:30, chess.ROOK:50, chess.QUEEN:90}.get(p.piece_type,10)
            # pawn shield reduces attacker effect modestly
            shield = 0
            if color == chess.WHITE:
                for df in (-1,0,1):
                    nf = kf + df; nr = kr + 1
                    if 0 <= nf <= 7 and 0 <= nr <= 7 and chess.square(nf,nr) in board.pieces(chess.PAWN, chess.WHITE):
                        shield += 1
            else:
                for df in (-1,0,1):
                    nf = kf + df; nr = kr - 1
                    if 0 <= nf <= 7 and 0 <= nr <= 7 and chess.square(nf,nr) in board.pieces(chess.PAWN, chess.BLACK):
                        shield += 1
            ks = defender_value - int(attacker_value * max(0.2, 1.0 - 0.15*shield))
            score += ks if color == chess.WHITE else -ks
        return score
