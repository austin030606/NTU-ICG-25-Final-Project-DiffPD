        import pickle
        import matplotlib.pyplot as plt
        from pathlib import Path
        import numpy as np


        file_path = Path("duck_3d")
        with open(file_path, "rb") as f:
            data = pickle.load(open(file_path / 'data_0008_threads.bin', 'rb'))

        method_name_map = {
            'pd_eigen_pcg_original': 'DiffPD',
            'pd_eigen_pcg_proj_abs': 'Abs Method',
            'pd_eigen_pcg_proj_clmp': 'Clamping Method',
            'newton_cholesky': 'Newton Cholesky'
        }

        method_keys = [k for k in data.keys() if isinstance(data[k], list) and len(data[k]) > 0 and isinstance(data[k][0], dict)]


        plt.figure(figsize=(15, 4.5))


        plt.subplot(1, 3, 1)
        for method in method_keys:
            records = data[method]
            grads = [np.linalg.norm(r['grad']) for r in records]
            label = method_name_map.get(method, method)
            plt.plot(grads, label=label)
        plt.xlabel("Iteration")
        plt.ylabel("Gradient Norm (|grad|)")
        plt.title("Gradient Norm per Iteration")
        plt.legend()
        plt.grid(True)


        plt.subplot(1, 3, 2)
        for method in method_keys:
            records = data[method]
            losses = [r['loss'] for r in records]
            label = method_name_map.get(method, method)
            plt.plot(losses, label=label)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss per Iteration")
        plt.legend()
        plt.grid(True)


        plt.subplot(1, 3, 3)
        for method in method_keys:
            records = data[method]
            total_times = [r['forward_time'] + r['backward_time'] for r in records]
            label = method_name_map.get(method, method)
            plt.plot(total_times, label=label)
        plt.xlabel("Iteration")
        plt.ylabel("Total Time (s)")
        plt.title("Total Time per Iteration")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        combined_path = Path("duck_3d/combined_duck_plots.png")
        plt.savefig(combined_path)

