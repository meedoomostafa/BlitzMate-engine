import chess
from engine.config import CONFIG

class Evaluator:
    def __init__(self):
        self.cfg = CONFIG.eval

    def evaluate(self, board: chess.Board) -> int:
        # Checking if draw exist
        if board.is_insufficient_material():
            return 0
        if board.is_stalemate():
            return 0
        if board.is_fivefold_repetition():
            return 0

        # Initialize Scores [MiddleGame, EndGame]
        mg_score = 0
        eg_score = 0
        phase = 0 

        # Cache commonly used data
        piece_map = board.piece_map()
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        # Main Loop (Material + PST)
        for sq, piece in piece_map.items():
            pt = piece.piece_type
            color = piece.color
            
            # Phase Calculation
            if pt == chess.KNIGHT: phase += 1
            elif pt == chess.BISHOP: phase += 1
            elif pt == chess.ROOK: phase += 2
            elif pt == chess.QUEEN: phase += 4
            
            # Material Values
            if pt == chess.PAWN:
                mg_val = self.cfg.MATERIAL_MG['PAWN']
                eg_val = self.cfg.MATERIAL_EG['PAWN']
            elif pt == chess.KNIGHT:
                mg_val = self.cfg.MATERIAL_MG['KNIGHT']
                eg_val = self.cfg.MATERIAL_EG['KNIGHT']
            elif pt == chess.BISHOP:
                mg_val = self.cfg.MATERIAL_MG['BISHOP']
                eg_val = self.cfg.MATERIAL_EG['BISHOP']
            elif pt == chess.ROOK:
                mg_val = self.cfg.MATERIAL_MG['ROOK']
                eg_val = self.cfg.MATERIAL_EG['ROOK']
            elif pt == chess.QUEEN:
                mg_val = self.cfg.MATERIAL_MG['QUEEN']
                eg_val = self.cfg.MATERIAL_EG['QUEEN']
            elif pt == chess.KING:
                mg_val = 0
                eg_val = 0
            
            # Piece Square Tables (PST)
            pst_sq = sq if color == chess.WHITE else chess.square_mirror(sq)
            
            # Fetch PST from Config (Safety check using getattr)
            pst_mg, pst_eg = 0, 0
            # Map piece type to config attribute names
            p_name = chess.piece_name(pt).upper() # e.g. "PAWN", "KNIGHT"
            
            # get PST_PIECE_MG / PST_PIECE_EG
            table_mg = getattr(self.cfg, f"PST_{p_name}_MG", None)
            table_eg = getattr(self.cfg, f"PST_{p_name}_EG", None)
            
            if table_mg and 0 <= pst_sq < 64: pst_mg = table_mg[pst_sq]
            if table_eg and 0 <= pst_sq < 64: pst_eg = table_eg[pst_sq]
            # Default penalty for minor pieces in EG if no table
            else: pst_eg = -10 if pt != chess.PAWN else 0 

            if color == chess.WHITE:
                mg_score += mg_val + pst_mg
                eg_score += eg_val + pst_eg
            else:
                mg_score -= mg_val + pst_mg
                eg_score -= eg_val + pst_eg


        # Structural Evaluation Part 
        
        # Bishop Pair Bonus
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            mg_score += self.cfg.BISHOP_PAIR_BONUS
            eg_score += self.cfg.BISHOP_PAIR_BONUS
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            mg_score -= self.cfg.BISHOP_PAIR_BONUS
            eg_score -= self.cfg.BISHOP_PAIR_BONUS

        # Pawn Structure
        mg_score += self._eval_pawns(board, white_pawns, chess.WHITE)
        mg_score -= self._eval_pawns(board, black_pawns, chess.BLACK)
        eg_score += self._eval_pawns(board, white_pawns, chess.WHITE)
        eg_score -= self._eval_pawns(board, black_pawns, chess.BLACK)

        # Search for threats
        threat_score = self._eval_threats(board, piece_map)
        mg_score += threat_score
        eg_score += threat_score

        # Tapered Evaluation Blending 
        phase = min(phase, 24)
        final_score = (mg_score * phase + eg_score * (24 - phase)) // 24
        
        if board.turn == chess.BLACK:
            return -final_score
        
        return final_score

    def _eval_pawns(self, board, pawns, color):
        score = 0
        for sq in pawns:
            file = chess.square_file(sq)
            rank = chess.square_rank(sq)
            
            # Isolated Pawn
            is_isolated = True
            if file > 0:
                for r in range(8):
                    if board.piece_at(chess.square(file-1, r)) == chess.Piece(chess.PAWN, color):
                        is_isolated = False; break
            if is_isolated and file < 7:
                 for r in range(8):
                    if board.piece_at(chess.square(file+1, r)) == chess.Piece(chess.PAWN, color):
                        is_isolated = False; break
            
            if is_isolated:
                score += self.cfg.ISOLATED_PAWN_PENALTY

            # Passed Pawn
            if self._is_passed(board, sq, color):
                relative_rank = rank if color == chess.WHITE else 7 - rank
                score += self.cfg.PASSED_PAWN_BONUS[relative_rank]
                
        return score

    def _is_passed(self, board, sq, color):
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        enemy_pawn = chess.Piece(chess.PAWN, not color)
        
        start_rank = rank + 1 if color == chess.WHITE else 0
        end_rank = 8 if color == chess.WHITE else rank
        
        for f in range(max(0, file-1), min(7, file+1) + 1):
            for r in range(start_rank, end_rank):
                if board.piece_at(chess.square(f, r)) == enemy_pawn:
                    return False
        return True

    def _eval_threats(self, board:chess.Board
                      , piece_map:dict[chess.Square,chess.Piece]):
        """
        Detects hanging pieces and bad trades.
        Returns score adjustment from White's perspective.
        """
        score = 0
        
        # Simple values for threat calculation (divide by 10 for "danger" scaling)
        values = {chess.PAWN: 10, chess.KNIGHT: 30, chess.BISHOP: 30
                  , chess.ROOK: 50, chess.QUEEN: 90, chess.KING: 100}
        
        for sq, piece in piece_map.items():
            if piece.piece_type == chess.KING:
                continue # King safety handled separately or by search (checkmate)
                
            attackers = board.attackers(not piece.color, sq)
            
            if attackers:
                if piece.piece_type > chess.PAWN:
                    for atk_sq in attackers:
                        attacker_piece = board.piece_at(atk_sq)
                        if attacker_piece and attacker_piece.piece_type == chess.PAWN:
                            penalty = 50 
                            score += penalty if piece.color == chess.BLACK else -penalty
                            break
                
                defenders = board.attackers(piece.color, sq)
                if len(attackers) > len(defenders):
                    val = values[piece.piece_type] * 5 
                    score += val if piece.color == chess.BLACK else -val
                
                elif len(defenders) == 0:
                    val = values[piece.piece_type] * 5
                    score += val if piece.color == chess.BLACK else -val

        return score