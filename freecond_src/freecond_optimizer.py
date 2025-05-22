import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

class GPRegression:
    def __init__(self, kernel_fn='rbf', length_scale=1.0, noise=0.1):
        """
        初始化高斯過程回歸模型
        
        Args:
            kernel_fn: 核函數類型，目前支援 'rbf'
            length_scale: RBF 核函數的長度尺度
            noise: 觀測噪聲的標準差
        """
        self.kernel_fn = kernel_fn
        self.length_scale = torch.nn.Parameter(torch.tensor(length_scale))
        self.noise = torch.nn.Parameter(torch.tensor(noise))
        self.X_train = None
        self.y_train = None
        self.K_inv = None
        
        # 為了支援多個輸出維度，創建標準化器
        self.X_scaler = StandardScaler()
        self.y_scalers = []
        
        # 超參數優化器
        self.params = [self.length_scale, self.noise]
    
    def rbf_kernel(self, X1, X2):
        """RBF (高斯) 核函數"""
        # 計算成對距離
        X1_norm = torch.sum(X1**2, dim=1).view(-1, 1)
        X2_norm = torch.sum(X2**2, dim=1).view(1, -1)
        dist = X1_norm + X2_norm - 2 * torch.mm(X1, X2.t())
        
        # 計算核函數值
        return torch.exp(-0.5 * dist / (self.length_scale**2))
    
    def compute_kernel(self, X1, X2):
        """計算核函數矩陣"""
        if self.kernel_fn == 'rbf':
            return self.rbf_kernel(X1, X2)
        else:
            raise ValueError(f"不支援的核函數: {self.kernel_fn}")
    
    def fit(self, X, y, n_iter=100, lr=0.01):
        """
        擬合高斯過程模型
        
        Args:
            X: 輸入特徵 [n_samples, n_features]
            y: 輸出標籤 [n_samples, n_outputs]
            n_iter: 超參數優化迭代次數
            lr: 學習率
        """
        # 資料標準化
        X_scaled = torch.tensor(self.X_scaler.fit_transform(X), dtype=torch.float32)
        
        # 記錄訓練資料
        self.X_train = X_scaled
        
        # 對每個輸出維度單獨處理
        n_outputs = y.shape[1]
        self.y_scalers = []
        y_scaled_list = []
        
        for i in range(n_outputs):
            scaler = StandardScaler()
            y_scaled = scaler.fit_transform(y[:, i].reshape(-1, 1))
            y_scaled_list.append(torch.tensor(y_scaled, dtype=torch.float32).view(-1))
            self.y_scalers.append(scaler)
            
        self.y_train = torch.stack(y_scaled_list, dim=1)
        
        # 設置優化器
        optimizer = torch.optim.Adam(self.params, lr=lr)
        
        # 優化核函數超參數
        for i in range(n_iter):
            optimizer.zero_grad()
            
            # 計算核函數矩陣
            K = self.compute_kernel(self.X_train, self.X_train)
            K = K + self.noise**2 * torch.eye(K.shape[0])
            
            # 計算負對數邊緣概率
            try:
                L = torch.linalg.cholesky(K)  # 計算 Cholesky 分解
                alpha = torch.cholesky_solve(self.y_train, L)
                
                # 損失函數：負對數邊緣概率
                loss = 0
                for j in range(n_outputs):
                    log_det = 2 * torch.sum(torch.log(torch.diag(L)))
                    quadratic = torch.mm(self.y_train[:, j].view(1, -1), alpha[:, j].view(-1, 1))
                    loss += 0.5 * (log_det + quadratic + K.shape[0] * torch.log(2 * torch.tensor(np.pi)))
                    
                # 反向傳播
                loss.backward()
                optimizer.step()
                
                if (i + 1) % 20 == 0:
                    print(f"Iteration {i+1}/{n_iter}, Loss: {loss.item():.4f}, "
                          f"length_scale: {self.length_scale.item():.4f}, noise: {self.noise.item():.4f}")
                    
            except RuntimeError:
                print("警告：核函數矩陣不是正定的，跳過此迭代")
                continue
        
        # 預先計算逆核函數矩陣，以加速預測
        K = self.compute_kernel(self.X_train, self.X_train)
        K = K + self.noise**2 * torch.eye(K.shape[0])
        self.K_inv = torch.linalg.inv(K)
        
        return self
        
    def predict(self, X, return_std=False):
        """
        預測新的輸入點的輸出值
        
        Args:
            X: 輸入特徵 [n_samples, n_features]
            return_std: 是否返回標準差
            
        Returns:
            預測的均值，如果 return_std=True，則也返回標準差
        """
        # 資料標準化
        X_scaled = torch.tensor(self.X_scaler.transform(X), dtype=torch.float32)
        
        # 計算測試點和訓練點之間的核函數
        K_s = self.compute_kernel(X_scaled, self.X_train)
        
        # 計算預測均值
        mean = torch.mm(K_s, torch.mm(self.K_inv, self.y_train))
        
        # 逆轉標準化
        mean_orig = np.zeros((X.shape[0], len(self.y_scalers)))
        for i in range(len(self.y_scalers)):
            mean_orig[:, i] = self.y_scalers[i].inverse_transform(
                mean[:, i].detach().numpy().reshape(-1, 1)
            ).flatten()
        
        if return_std:
            # 計算核函數矩陣 K(X_*, X_*)
            K_ss = self.compute_kernel(X_scaled, X_scaled)
            
            # 計算預測方差
            var = K_ss - torch.mm(K_s, torch.mm(self.K_inv, K_s.t()))
            var = torch.diag(var)
            
            # 轉換回原始尺度
            std_orig = np.zeros((X.shape[0], len(self.y_scalers)))
            for i in range(len(self.y_scalers)):
                # 只考慮方差的縮放，忽略均值偏移
                scale = self.y_scalers[i].scale_[0]
                std_orig[:, i] = np.sqrt(var.detach().numpy()) * scale
                
            return mean_orig, std_orig
        
        return mean_orig


class FCOptimizer:
    def __init__(self, metric_list, kernel_fn='rbf', length_scale=1.0, noise=0.1, pretrained_path=None):
        self.metric_list = metric_list
        self.metric2index = {metric: i for i, metric in enumerate(metric_list)}
        self.gp_model = GPRegression(kernel_fn=kernel_fn, length_scale=length_scale, noise=noise)
        if pretrained_path:
            self.load(pretrained_path)
        self.trained = False

    def train(self, X_train, y_train, n_iter=00, lr=0.01):
        self.gp_model.fit(X_train, y_train, n_iter=n_iter, lr=lr)
        self.trained = True

    def get_objective_score_gp(self, input_features, objective):
        input_np = np.array([input_features])
        predictions = self.gp_model.predict(input_np)[0]
        score = 0
        for metric, weight in objective.items():
            if metric in self.metric2index:
                index = self.metric2index[metric]
                score += predictions[index] * weight
        return score

    def optimize(self, objective, space=None, n_calls=50, n_random_starts=10, random_state=42, verbose=False):
        if space is None:
            space = [
                Real(1, 4.0, name='fg'),
                Real(0.0, 1.0, name='bg'),
                Real(8.0, 32.0, name='fqth')
            ]
        @use_named_args(space)
        def objective_function_gp(fg, bg, fqth):
            score = self.get_objective_score_gp([fg, bg, fqth], objective)
            return -score  # minimize

        result = gp_minimize(
            objective_function_gp,
            space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            random_state=random_state,
            verbose=verbose,
            n_jobs=-1
        )
        best_params = result.x
        best_score = -result.fun
        best_input = np.array([best_params])
        best_output = self.gp_model.predict(best_input)[0]
        return {
            "best_params": {
                "fg": best_params[0],
                "bg": best_params[1],
                "fqth": best_params[2]
            },
            "best_score": best_score,
            "best_output": {metric: best_output[i] for metric, i in self.metric2index.items()},
            "result_obj": result
        }

    def save(self, path):
        # 儲存 GPRegression 參數與 scaler
        with open(path, 'wb') as f:
            pickle.dump({
                "gp_state": self.gp_model.__dict__,
                "metric_list": self.metric_list
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.gp_model.__dict__.update(data["gp_state"])
            self.metric_list = data["metric_list"]
            self.metric2index = {metric: i for i, metric in enumerate(self.metric_list)}
            self.trained = True

if __name__ == "__main__":
    import os
    import pandas as pd
    from sklearn.model_selection import train_test_split

    def parse_result(csv_path):
        df = pd.read_csv(csv_path, header=None)
        df = df.iloc[1:]
        gen_result = dict(zip(df[0], df[1]))

        return gen_result
    def para2_features(para_str):
        """
        從參數字符串中提取 fg, bg, fqth 的值
        
        Args:
            para_str: 格式類似 'cfg_15_fg_2.215_1_bg_0.295_0_fqth_21.560000000000002_tfc_25'
        
        Returns:
            包含特徵值的字典
        """
        features = {}
        
        # 提取 fg 值
        if '_fg_' in para_str:
            parts = para_str.split('_fg_')[1].split('_bg_')[0].split('_')
            features['fg_1'] = float(parts[0])
            features['fg_2'] = float(parts[1]) if len(parts) > 1 else features['fg_1']
        
        # 提取 bg 值
        if '_bg_' in para_str:
            parts = para_str.split('_bg_')[1].split('_fqth_')[0].split('_')
            features['bg_1'] = float(parts[0])
            features['bg_2'] = float(parts[1]) if len(parts) > 1 else features['bg_1']
        
        # 提取 fqth 值
        if '_fqth_' in para_str:
            parts = para_str.split('_fqth_')[1].split('_tfc_')[0]
            features['fqth'] = float(parts)
        indicator = False
        if features["fg_1"]!=features["fg_2"] or features["bg_1"]!=features["bg_2"]:
            indicator = True

        return [features['fg_1'], features['bg_1'], features['fqth']], indicator
    def result2_features(result_dict,metric_list):
        """
        將結果字典轉換為特徵列表
        
        Args:
            result_dict: 包含評估指標的字典
            metric_list: 評估指標列表
        
        Returns:
            特徵列表
        """
        features = []
        for metric in metric_list:
            if metric in result_dict:
                features.append(result_dict[metric])
            else:
                raise ValueError(f"Metric '{metric}' not found in result dictionary.")
                features.append(0)  # 如果指標不存在，則填充0
        return features



    meta_dir = "runs/data/FCIP_full/sd"

    para2results={}

    for df_dir in os.listdir(meta_dir):
        for file in os.listdir(os.path.join(meta_dir, df_dir)):
            if file == "evaluation_result_sum.csv":
                file_path = os.path.join(meta_dir, df_dir, file)
                #print(f"Parsing ./{file_path}")
                result = parse_result(file_path)
                para2results[df_dir] = result 


    metric_list=[
        "Image Reward",
        "HPS V2.1",
        #"Aesthetic Score",
        "PSNR",
        "LPIPS",
        #'MSE',
        "CLIP Similarity",
        "IoU Score",
        #'T2V Score',
        "DINO",
    ]


    input_list=[]
    output_list=[]
    for para, result in para2results.items():
        input_features, indicator = para2_features(para)
        if indicator:
            pass
        else:
            input_list.append(input_features)
            output_list.append(result2_features(result, metric_list))
    print(f"input_list shape: {len(input_list)}, output_list shape: {len(output_list)}")
    # 將輸入和輸出轉換為PyTorch張量
    input_array = np.stack(input_list)
    output_array = np.stack(output_list)

    optimizer = FCOptimizer(metric_list)
    optimizer.train(input_array, output_array)
    optimizer.save("gp_model.pkl")

    optimizer = FCOptimizer(metric_list, pretrained_path="gp_model.pkl")
    result = optimizer.optimize({
        "HPS V2.1": 100,
        "CLIP Similarity": 1.5,
    })
    print(result["best_params"])
    print(result["best_score"])
    print(result["best_output"])