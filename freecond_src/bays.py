import numpy as np
import pandas as pd
import GPy
import GPyOpt
from GPyOpt.core.task.space import Design_space
from GPyOpt.experiment_design.latin_design import (
    LatinDesign,
)  # Import the correct class
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle


class SurrogateModelTrainer:
    def __init__(self, param_bounds, output_names, sample_size=100):
        """
        初始化 Surrogate Model 訓練器

        Args:
            param_bounds (dict): 參數範圍，例如 {'fg': (0.0, 10.0), 'bg': (0.0, 10.0), 'fqth': (0.0, 10.0)}
            output_names (list): 輸出指標名稱，例如 ['clip', 'iou', 'lpips', 'imagereward']
            sample_size (int): 初始採樣點數量
        """
        self.param_bounds = param_bounds
        self.params = list(param_bounds.keys())
        self.output_names = output_names
        self.sample_size = sample_size
        self.models = {}
        self.data = None

    def plot_model_predictions(self):
        """繪製模型預測圖表，用於視覺化評估"""
        if not self.models or self.data is None:
            print("請先訓練模型")
            return

        X = self.data[self.params].values

        for output_name, model in self.models.items():
            y_actual = self.data[output_name].values
            y_pred, y_var = model.predict(X)

            plt.figure(figsize=(10, 6))
            plt.scatter(y_actual, y_pred, alpha=0.6)
            plt.plot(
                [y_actual.min(), y_actual.max()],
                [y_actual.min(), y_actual.max()],
                "k--",
            )
            plt.fill_between(
                np.sort(y_actual.flatten()),
                np.sort(y_pred.flatten() - 1.96 * np.sqrt(y_var.flatten())),
                np.sort(y_pred.flatten() + 1.96 * np.sqrt(y_var.flatten())),
                alpha=0.2,
            )
            plt.xlabel(f"實際 {output_name}")
            plt.ylabel(f"預測 {output_name}")
            plt.title(f"{output_name} 預測 vs. 實際值")
            plt.grid(True, alpha=0.3)
            plt.show()

    @classmethod
    def load_models(cls, filepath):
        """載入已訓練好的模型"""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        trainer = cls(data["param_bounds"], data["output_names"])
        print("# load trainer parameters")
        print(f"  param_bounds: {trainer.param_bounds}")
        print(f"  output_names: {trainer.output_names}")
        trainer.models = data["models"]
        return trainer


class SurrogateOptimizer:
    def __init__(self, surrogate_trainer_or_path):
        """
        初始化 Surrogate 優化器

        Args:
            surrogate_trainer_or_path: 已訓練好的 SurrogateModelTrainer 對象或模型文件路徑
        """
        if isinstance(surrogate_trainer_or_path, str):
            self.trainer = SurrogateModelTrainer.load_models(surrogate_trainer_or_path)
        else:
            self.trainer = surrogate_trainer_or_path

        self.models = self.trainer.models
        self.param_bounds = self.trainer.param_bounds
        self.params = list(self.param_bounds.keys())
        self.output_names = self.trainer.output_names

    def predict(self, fg, bg, fqth):
        """使用 surrogate models 預測給定參數的評估指標"""
        X = np.array([[fg, bg, fqth]])

        predictions = {}
        for output_name, model in self.models.items():
            y_pred, y_var = model.predict(X)
            predictions[output_name] = (float(y_pred[0][0]), float(y_var[0][0]))

        return predictions

    def _objective_function(self, X, weights=None, constraints=None):
        """
        優化目標函數

        Args:
            X (numpy.ndarray): 參數值
            weights (dict): 各指標的權重，默認為等權重
            constraints (dict): 約束條件，例如 {'lpips': (-float('inf'), 0.1)}

        Returns:
            float: 加權目標函數值（負值，因為 GPyOpt 默認最小化）
        """
        fg, bg, fqth = X[0]
        predictions = self.predict(fg, bg, fqth)
        # print("_objective predictions", predictions)
        # 默認權重
        if weights is None:
            weights = {name: 1.0 for name in self.output_names}

        # 計算加權和
        weighted_sum = 0
        for name, (pred, _) in predictions.items():
            if name in weights:
                weighted_sum += weights[name] * pred

        # 檢查約束條件
        if constraints:
            for name, (min_val, max_val) in constraints.items():
                if name in predictions:
                    pred_val = predictions[name][0]
                    if pred_val < min_val or pred_val > max_val:
                        return -1e6  # 違反約束，返回大的懲罰值

        # 返回負值用於最小化（GPyOpt默認最小化）
        # print(f"  weighted_sum: {weighted_sum:.4f}")
        return -weighted_sum

    def optimize(
        self,
        weights=None,
        constraints=None,
        acquisition_type="LCB",
        max_iter=20,
        batch_size=4,
        num_cores=8,
        verbose=True,
        report_file=None,
        evaluations_file=None,
    ):
        """
        使用貝葉斯優化尋找最佳參數

        Args:
            weights (dict): 各指標的權重，例如 {'clip': 5.0, 'iou': 2.0, 'lpips': 1.0}
            constraints (dict): 約束條件，例如 {'lpips': (-float('inf'), 0.1)}
            max_iter (int): 最大迭代次數
            verbose (bool): 是否顯示詳細信息
            report_file (str): 儲存報告的文件路徑 (bays_optimize_report.txt)
            evaluations_file (str): 儲存評估結果的文件路徑 (bays_optimize_evaluations.csv)

        Returns:
            tuple: (最佳參數, 預測效果)
        """
        # 定義參數空間
        param_space = []
        for param, (lower, upper) in self.param_bounds.items():
            param_space.append(
                {"name": param, "type": "continuous", "domain": (lower, upper)}
            )

        # 定義目標函數
        def objective(X):
            return self._objective_function(X, weights, constraints)

        # 初始化貝葉斯優化器
        print("Initializing Bayesian Optimization...")
        print(f"  acquisition_type: {acquisition_type}")
        optimizer = GPyOpt.methods.BayesianOptimization(
            f=objective,
            domain=param_space,
            model_type="GP",
            acquisition_type=acquisition_type,
            normalize_Y=True,
            maximize=False,
            verbosity=False,
            batch_size=batch_size,
            num_cores=num_cores,
        )

        # 運行優化
        optimizer.run_optimization(
            max_iter=max_iter,
            verbosity=False,
            report_file=report_file,
            evaluations_file=evaluations_file,
        )

        # 獲取最佳參數
        best_params = optimizer.x_opt
        opt_1, opt_2, opt_3 = best_params

        # 使用 surrogate model 預測最佳效果
        best_predictions = self.predict(opt_1, opt_2, opt_3)

        # 輸出結果
        if verbose:
            print("\nBest param:")
            print(f"  fg: {opt_1:.4f}")
            print(f"  bg: {opt_2:.4f}")
            print(f"  fqth: {opt_3:.4f}")
            print("\nPrediction:")
            for name, (pred, var) in best_predictions.items():
                print(f"  {name}: {pred:.4f} ± {np.sqrt(var):.4f}")

        return (opt_1, opt_2, opt_3), best_predictions
