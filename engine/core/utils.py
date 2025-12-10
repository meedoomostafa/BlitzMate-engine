def print_info(d, score, nodes, elapsed, pv_moves, ponder, MATE_SCORE):
        pv_str = " ".join([m.uci() for m in pv_moves])
        ponder_str = ponder.uci() if ponder else "-"
        nps = int(nodes / elapsed) if elapsed > 0 else 0
        
        if abs(score) > MATE_SCORE - 100:
            mate_in = (MATE_SCORE - abs(score) + 1) // 2
            score_str = f"mate {mate_in if score > 0 else -mate_in}"
        else:
            score_str = f"cp {score}"

        print(f"info depth {d} score {score_str} nodes {nodes} nps {nps} time {int(elapsed)} ponder {ponder_str} pv {pv_str}")