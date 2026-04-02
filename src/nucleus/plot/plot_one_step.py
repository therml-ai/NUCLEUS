import matplotlib.pyplot as plt
import seaborn as sns
import torch

def pretty_name(case_name: str):
    parts = case_name.split("_")
    parts[0] = parts[0].capitalize()[:3]
    parts[1] = parts[1].upper()
    parts[2] = str(int(float(parts[2])))
    parts.append("°C")
    return " ".join(parts)

def filter_sat(test_results):
    return [t for t in test_results if "sub" in t.case_name]

def latex():
    checkpoints = {
        "Neighbor MoE": "/pub/afeeney/bubbleformer_logs/neighbor_moe_poolboiling64_48105352/checkpoints/inference_rollouts/test_one_step.pt",
        "Neighbor MLP": "/pub/afeeney/bubbleformer_logs/neighbor_vit_poolboiling64_48103654/checkpoints/inference_rollouts/test_one_step.pt",
        "Axial MoE": "/pub/afeeney/bubbleformer_logs/axial_moe_poolboiling64_48103671/checkpoints/inference_rollouts/test_one_step.pt",
        "Axial MLP": "/pub/afeeney/bubbleformer_logs/axial_vit_poolboiling64_48103668/checkpoints/inference_rollouts/test_one_step.pt",
        "Global MoE": "/pub/afeeney/bubbleformer_logs/vit_moe_poolboiling64_48103688/checkpoints/inference_rollouts/test_one_step.pt",
        "Global MLP": "/pub/afeeney/bubbleformer_logs/vit_poolboiling64_48103690/checkpoints/inference_rollouts/test_one_step.pt",
        "BubbleFormer 1": "/pub/afeeney/bubbleformer_logs/bubbleformer_film_vit_poolboiling64_48105033/checkpoints/inference_rollouts/test_one_step.pt",
        "BubbleFormer 2": "/pub/afeeney/bubbleformer_logs/bubbleformer_film_vit_poolboiling64_48105034/checkpoints/inference_rollouts/test_one_step.pt",
    }

    print(r"\begin{table}")
    print(r"\centering")
    print(r"\begin{tabular}" + "{" + "c|" + "c"*14 + "}")
    for col, (name, checkpoint) in enumerate(checkpoints.items()):
        test_results = torch.load(checkpoint, weights_only=False)
        test_results = filter_sat(test_results)
        
        case = [pretty_name(t.case_name) for t in test_results]
        sdf = [t.sdf_mae for t in test_results]
        temp = [t.temp_mae for t in test_results]
        velx = [t.velx_mae for t in test_results]
        vely = [t.vely_mae for t in test_results]
        mae = [t.mae for t in test_results]
        
        sum_sdf = sum(sdf)
        sum_temp = sum(temp)
        sum_velx = sum(velx)
        sum_vely = sum(vely)
        sum_mae = sum(mae)
        
        print(r"\hline")
        print(" & & Total & " + " & ".join(case) + " \\\\")
        print(r"\hline")
        print(r"\multirow{5}{*}" + "{" + name + "}" + " & SDF " + f" & {sum_sdf:.4f} &" + " & ".join([f"{s:.4f}" for s in sdf]) + r"\\")
        print(r" & Temp " + f" & {sum_temp:.4f} &" + " & ".join([f"{s:.4f}" for s in temp]) + r"\\")
        print(r" & Velx " + f" & {sum_velx:.4f} &" + " & ".join([f"{s:.4f}" for s in velx]) + r"\\")
        print(r" & Vely " + f" & {sum_vely:.4f} &" + " & ".join([f"{s:.4f}" for s in vely]) + r"\\")
        print(r" & MAE " + f" & {sum_mae:.4f} &" + " & ".join([f"{s:.4f}" for s in mae]) + r"\\")
        
    print(r"\end{tabular}")
    print(r"\end{table}")
    
latex()