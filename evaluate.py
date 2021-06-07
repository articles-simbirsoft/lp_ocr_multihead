import numpy as np
import torch
import torch.nn.functional as F

def evaluate_model(model, dataset, label_encoder):
    model.eval()

    results = []
    with torch.no_grad():
        for img, label in dataset:
            preds = model(torch.tensor(img).unsqueeze(0).cuda())
            preds = [
                torch.argmax(F.softmax(p, dim=1)).squeeze(0).detach().cpu().numpy()
                for p in preds
            ]

            preds = label_encoder.decode(preds)
            label = label_encoder.decode(label)
            res = preds == label
            results.append(res)

            if not res:
                print(f"Incorrect prediction: {''.join(preds)} instead of {''.join(label)}")
    
    acc = np.sum(results) / len(results)
    print(f"Accuracy: {acc:.2f}")
