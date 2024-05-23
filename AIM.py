# 라이브러리 불러오기

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split


#%% 데이터 불러오기

os.chdir(os.path.dirname(os.path.realpath(__file__))) 
df = pd.read_csv('Data.csv', index_col= 0)
df = df.dropna()

# 훈련, 보정, 검증 기간 설정
calibration_start_period = '01.07.2022 07:50'
calibration_end_period = '24.08.2022 11:30'

# 대상 변수 지정
# =============================================================================
target_variable = 'WB_evaporator inlet temp.'
ground_truth = 'R_evaporator inlet temp.'
# =============================================================================

#%% 이상치 제거함수 선언 (데이터 전처리)

def Remove_outliers(df, threshold=3):
    
    from scipy.stats import zscore
    z_scores = np.abs(zscore(df))

    outliers_mask = (z_scores > threshold).any(axis=1)

    df_no_outliers = df[~outliers_mask]

    total_data_before = len(df)
    total_data_after = len(df_no_outliers)
    percentage_lost = ((total_data_before - total_data_after) / total_data_before) * 100

    # print(f"Total data before removal: {total_data_before}")
    # print(f"Total data after removal: {total_data_after}")
    print(f"Percentage of data lost: {percentage_lost:.2f}%")

    return df_no_outliers

#%% 정규화 함수 선언 (데이터 전처리)

def Normboundary_gen(df_):
    norm_bound = np.zeros((np.shape(df_)[1],2))
    for i in range(0,np.shape(df_)[1]):
        #각 변수 별 최대 최소 값을 산출해서 행렬로 저장
        norm_bound[[i],0] = [[df_[:,i].min()]]
        norm_bound[[i],1] = [[df_[:,i].max()]]        
    return norm_bound

def Normalize(df_,bound):
    df_norm = np.zeros(np.shape(df_))
    
    for i in range(0,np.shape(df_)[1]):
          # [0,1] 범위로 정규화 계산을 진행
        df_norm[:,[i]] = (df_[:,[i]]-bound[[i],0]) / (bound[[i],1]-bound[[i],0])
    print("Data Normalization completed!")    
    return df_norm

#%% 데이터 전처리

df_ = Remove_outliers(df, threshold=5) # 이상치 제거

# Min-max 정규화 및 정규화 경계 저장
Normalization_boundary = Normboundary_gen(np.array(df_))
Normalization_boundary = pd.DataFrame(Normalization_boundary,index = df_.columns)
Normalization_boundary.to_csv('Normalization_boundary.csv',index=True)

df_ = pd.DataFrame(Normalize(np.array(df_),np.array(Normalization_boundary)), columns = df_.columns, index = df_.index)
df_ = df_.dropna(axis=1)

# 원본 데이터
df_train = df.loc[:calibration_start_period]
df_calibration = df.loc[calibration_start_period:calibration_end_period]
df_test = df.loc[calibration_end_period:]

# 전처리 데이터
df_train_ = df_.loc[:calibration_start_period]
df_calibration_ = df_.loc[calibration_start_period:calibration_end_period]
df_test_ = df_.loc[calibration_end_period:]

#%% 대상 변수를 입력변수로하는 목적 변수 선택

def Correlation_Filter(df, target_variable, correlation_threshold):
    
    correlation_matrix = df.corr()
    correlation_matrix_ = correlation_matrix.loc[:, target_variable]
    sorted_correlations = correlation_matrix_.abs().sort_values(ascending=False)

    # 상관관계가 1인 값은 제외
    filtered_correlation = sorted_correlations[(sorted_correlations != 1) & (sorted_correlations.index != ground_truth)]
    selected_columns = list(filtered_correlation.index[0:3])  # 상관계수가 높은 상위 N개 변수 선택

    for i in range(3, len(filtered_correlation)):
        if filtered_correlation[i] > correlation_threshold:
            selected_columns.append(filtered_correlation.index[i])
        if i >= 7:  # 상관관계 임계값
            break

    correlation_df = pd.DataFrame(filtered_correlation[selected_columns])
    
    return selected_columns, correlation_df
        
correlation_output, correlation_df = Correlation_Filter(df_, target_variable, 0.7)

def Plot_Correlation(df, title):
    plt.rcParams['figure.figsize'] = [15, 15]
    plt.rcParams['font.family'] = 'times new roman'
    plt.rc('axes', labelsize=25)
    plt.rc('axes', titlesize=35)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    plt.rc('legend', fontsize=20)
    plt.rc('figure', titlesize=30)
    
    fig = plt.figure(figsize=(8, 6))
    
    ax = fig.add_subplot()
    ax.plot(df, marker='o', color='black')
    
    ax = plt.gca()
    xticks = ax.get_xticks()
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(df.index, rotation=90)

    fig.suptitle(title)   
    plt.show()
    
Plot_Correlation(correlation_df, 'Pearson correlation for selected ouput variables')

#%% 입력변수 선택

# 모델 정의
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

# 상위 10개의 특성 저장을 위한 데이터프레임 초기화
df_input_feature = pd.DataFrame(columns=['Y_variable', 'X_Variables'])

for i in tqdm(range(len(correlation_output)), desc='Selecting input features'):
   
    y = df_train_.loc[:, [correlation_output[i]]]
    X = df_train_.drop(columns=[target_variable, ground_truth,correlation_output[i]])
    
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    input_dim = X_tensor.shape[1]
    output_dim = y_tensor.shape[1]
    model_linear = LinearRegressionModel(input_dim, output_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_linear.parameters(), lr=0.1)
    
    num_epochs = 1000
    for epoch in range(num_epochs):
        # 순전파
        outputs = model_linear(X_tensor)
        loss = criterion(outputs, y_tensor)
        
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    learned_weights = model_linear.linear.weight.data.cpu().detach().numpy().flatten()
    
    # 가중치가 높은 상위 N개의 컬럼 선택
    top_indices = learned_weights.argsort()[-6:][::-1]
    top_features = X.columns[top_indices]
  
    # 결과 저장
    df_input_feature = df_input_feature.append({
        'Y_variable': correlation_output[i],
        'X_Variables': top_features.tolist() + [target_variable]
    }, ignore_index=True)

# print(df_input_feature)
#%% VIF 테스트

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_results = []

# tqdm을 사용하여 반복 작업의 진행 상황 표시
for candi in tqdm(range(len(df_input_feature))):
    
    # VIF 계산
    inputs_candidates = df_input_feature.loc[candi,'X_Variables']
    df_VIF = df_train_.loc[:, inputs_candidates]
    
    # 상수 항 추가
    df_VIF = sm.add_constant(df_VIF)
    vif_result = pd.DataFrame()
    while True:

        vif_result["Variables"] = df_VIF.columns
        vif_result["VIF"] = [variance_inflation_factor(df_VIF.values, i) for i in range(df_VIF.shape[1])]

        max_vif = vif_result[(vif_result['Variables'] != 'const') & (vif_result['Variables'] != target_variable)]["VIF"].max()
        if max_vif > 10:
            # Find the variable with the highest VIF and drop it from the DataFrame
            max_vif_var = vif_result.loc[vif_result["VIF"] == max_vif, "Variables"].values[0]
            df_VIF.drop(max_vif_var, axis=1, inplace=True)
            vif_result.drop(vif_result[vif_result["Variables"] == max_vif_var].index, inplace=True)
        else:
            break
    vif_results.append(vif_result)

print('\n')
print("------------------------------------------------------------")
print('Candiate for input variables of benchmark model is selected!')
print("------------------------------------------------------------")

for i in range(len(vif_results)):
    vif_results[i] = vif_results[i].drop(index=vif_results[i][(vif_results[i]['VIF'] == 0) | vif_results[i]['VIF'].isna()].index)
    vif_results[i] = vif_results[i].drop(index=vif_results[i].index[0])


#%% 예측과 데이터 비교 그래프 Plotting

def Do_scatter(pred, true, title):
    plt.rcParams['font.family']='times new roman' 
    plt.rcParams['figure.figsize']=[7,18]
    plt.rc('axes',labelsize = 25)
    plt.rc('axes',titlesize = 25)
    plt.rc('xtick',labelsize = 25)
    plt.rc('ytick',labelsize = 25)
    plt.rc('legend', fontsize= 20)
    plt.rc('figure',titlesize = 25)
    cols = pred.columns
    f, a = plt.subplots(1, pred.shape[1], dpi=720)
    
    from sklearn.metrics import r2_score
    R2 = r2_score(true,pred)
    
    for idx, col in enumerate(cols):
        pred_, true_ = pred[col], true[col]
        ax = a[idx] if pred.shape[1] > 1 else a
        ax.scatter(pred_, true_, s=5, color='k',alpha = 1)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xlim = [max(xmin, ymin), min(xmax, ymax)]

        ax.plot([max(xmin, ymin)*0.9, min(xmax, ymax)*1.1], [max(xmin, ymin)*0.9, min(xmax, ymax)*1.1], color='r', ls='--', zorder=100)
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
        ax.set_xticks(np.linspace(xlim[0], xlim[1], 3).round(1))
        ax.set_yticks(np.linspace(xlim[0], xlim[1], 3).round(1))
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground truth')
        
    plt.subplots_adjust(wspace=.7, bottom=.6)
    plt.suptitle(f'{title}', fontsize=30,x=0.5, y=0.915)
    plt.show()
    
    return f, a

#%% MLP 모델링

# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, l2_lambda=0.001):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        # 각 선형 레이어에 대한 L2 정규화를 적용
        self.l2_lambda = l2_lambda

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 모델 생성 함수
def create_mlp_model(input_shape):
    input_size = input_shape[1]
    hidden_size = 64  # 은닉층의 뉴런 수 (조정 가능)
    output_size = 1  # 출력 변수의 개수
    
    model = MLP(input_size, hidden_size, output_size).to(device)
    return model

# 학습 데이터를 PyTorch Tensor로 변환
def prepare_data(input_np, output_np):
    input_variables_tensor = torch.tensor(input_np, dtype=torch.float32).to(device)
    output_variable_tensor = torch.tensor(output_np, dtype=torch.float32).to(device)
    return input_variables_tensor, output_variable_tensor

# 학습 함수
def train_model(model, input_train, output_train, num_epochs=20, batch_size=32, validation_split=0.15):
    criterion = nn.MSELoss()  # 손실 함수: 평균 제곱 오차
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    dataset = TensorDataset(input_train, output_train)
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation loss 계산
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)
        
        # print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

# 모델 평가 함수
def evaluate_model(model, input_test, output_test):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(input_test, dtype=torch.float32).to(device)
        outputs = model(inputs)
        predictions = outputs.cpu().numpy()
    return predictions

# 모델 학습 및 평가
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from sklearn.metrics import mean_squared_error

result_df = pd.DataFrame(columns=['Output_Variable', 'MSE', 'Std'])
data_list = []
result_df = pd.DataFrame()

for i in tqdm(range(len(vif_results)), desc='MLP modeling process'):
    input_variables = list(vif_results[i].iloc[:,0])
    output_variable = list(correlation_df.index)[i]
    
    data_list.append({'Input_Variables': input_variables, 'Output_Variable': output_variable})
    
    input_df = df_train_.loc[:,input_variables]
    output_df = df_train_.loc[:,[output_variable]]
    
    input_np = np.array(input_df)
    output_np = np.array(output_df)
    
    input_train, input_test, output_train, output_test = train_test_split(input_np, output_np, test_size=0.2, shuffle=True, random_state=0)
    
    model = create_mlp_model(input_train.shape)
    input_train_tensor, output_train_tensor = prepare_data(input_train, output_train)
    
    train_model(model, input_train_tensor, output_train_tensor)
    
    predictions = evaluate_model(model, input_test, output_test)
    Do_scatter(pd.DataFrame(output_test), pd.DataFrame(predictions),output_variable)
    
    
    euclidean_distance = np.sqrt(np.sum((output_test - predictions) ** 2, axis=1))
    
    # Calculate MSE and standard deviation
    mse_to_y_equals_x = mean_squared_error(np.zeros_like(output_test), euclidean_distance)
    std_to_y_equals_x = np.std(euclidean_distance)
    
    result_df = result_df.append({'Output_Variable': output_variable, 
                                  'MSE': mse_to_y_equals_x, 
                                  'Std': std_to_y_equals_x}, 
                                 ignore_index=True)
    

    torch.save(model.state_dict(), f"model_{output_variable}.pth")

result_df.set_index('Output_Variable', inplace=True)
df_info = pd.DataFrame(data_list)

selected_benchmark_model = result_df['Std'].idxmin()
selected_input_variables = df_info[df_info['Output_Variable'] == selected_benchmark_model]['Input_Variables'].values[0]

#%%

input_test_ = df_test_.loc[:,selected_input_variables].to_numpy()
output_test_ = df_test_.loc[:,[selected_benchmark_model]].to_numpy()

input_test_tensor, output_test_tensor = prepare_data(input_test, output_test)

model = create_mlp_model(input_test.shape)
model.load_state_dict(torch.load('model_'+selected_benchmark_model+'.pth'))
model.eval()  # Set the model to evaluation mode after loading

predictions_test = evaluate_model(model, input_test_, output_test_)

Do_scatter(pd.DataFrame(output_test_), pd.DataFrame(predictions_test),selected_benchmark_model)










