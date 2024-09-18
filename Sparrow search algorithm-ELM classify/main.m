%_________________________________________________________________________%
% 基于麻雀优化ELM分类问题求解        %
%_________________________________________________________________________%
clear all 
clc
%% 导入数据
data=xlsread('tdata','all data','B2:E51');
Train = xlsread('tdata','training set','B2:E41');
Test = xlsread('tdata','test set','B2:E11');
% 训练数据
P_train = xlsread('tdata','training set','B2:D41')';
T_train = xlsread('tdata','training set','E2:E41')';
% 测试数据
P_test = xlsread('tdata','test set','B2:D11')';
T_test = xlsread('tdata','test set','E2:E11')';
%% 以上都改

%% ------采用SSA—ELM进行分类----------------------------------------- 
%训练数据相关尺寸
R = size(P_train,1);
S = size(T_train,1);
N = 100;%隐含层个数
%% 定义麻雀优化参数
pop=20; %种群数量
Max_iteration=200; %  设定最大迭代次数
dim = N*R + N;%维度，即权值与阈值的个数
lb = [-1.*ones(1,N*R),zeros(1,N)];%下边界
ub = [ones(1,N*R),ones(1,N)];%上边界
fobj = @(x) fun(x,P_train,T_train,N,P_test,T_test);
[Best_pos,Best_score,SSA_curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj); %开始优化
[fitness,IW,B,LW,TF,TYPE] = fun(Best_pos,P_train,T_train,N,P_test,T_test);%获取优化后的相关参数
figure
plot(SSA_curve,'linewidth',1.5);
grid on
xlabel('迭代次数')
ylabel('适应度函数')
title('SSA-ELM迭代曲线')
%% ELM仿真测试
T_sim_1 = elmpredict(P_train,IW,B,LW,TF,TYPE);
T_sim_2 = elmpredict(P_test,IW,B,LW,TF,TYPE);

%% 结果对比
disp('SSA-ELM结果展示：----------------')
result_1 = [T_train' T_sim_1'];
result_2 = [T_test' T_sim_2'];
% 训练集正确率
k1 = length(find(T_train == T_sim_1));
n1 = length(T_train);
Accuracy_1 = k1 / n1 * 100;
disp(['训练集正确率Accuracy = ' num2str(Accuracy_1) '%(' num2str(k1) '/' num2str(n1) ')'])
% 测试集正确率
k2 = length(find(T_test == T_sim_2));
n2 = length(T_test);
Accuracy_2 = k2 / n2 * 100;
disp(['测试集正确率Accuracy = ' num2str(Accuracy_2) '%(' num2str(k2) '/' num2str(n2) ')'])
%% 显示
count_A = length(find(T_train == 1));
count_B = length(find(T_train == 2));
count_C = length(find(T_train == 3));
rate_A = count_A / 50;
rate_B = count_B / 50;
rate_C = count_C / 50;
total_A = length(find(data(:,4) == 1));
total_B = length(find(data(:,4) == 2));
total_C = length(find(data(:,4) == 3));
number_A = length(find(T_test == 1));
number_B = length(find(T_test == 2));
number_C = length(find(T_test == 3));
number_A_sim = length(find(T_sim_2 == 1 & T_test == 1));
number_B_sim = length(find(T_sim_2 == 2 & T_test == 2));
number_C_sim = length(find(T_sim_2 == 3 & T_test == 3));
disp(['鸢尾花总数：' num2str(50)...
      '  山鸢尾：' num2str(total_A)...
      '  彩鸢尾：' num2str(total_B)...
      '  维吉尼亚鸢尾：' num2str(total_C)]);
disp(['训练集鸢尾花总数：' num2str(40)...
      '  山鸢尾：' num2str(count_A)...
      '  彩鸢尾：' num2str(count_B)...
      '  维吉尼亚鸢尾：' num2str(count_C)]);
disp(['测试集鸢尾花总数：' num2str(10)...
      '  山鸢尾：' num2str(number_A)...
      '  彩鸢尾：' num2str(number_B)...
      '  维吉尼亚鸢尾：' num2str(number_B)]);
disp(['山鸢尾测试集预测正确数： ' num2str(number_A_sim)...
      '山鸢尾测试集预测错误数： ' num2str(number_A - number_A_sim)...
      '山鸢尾测试集预测正确率=' num2str(number_A_sim/number_A*100) '%']);
disp(['彩鸢尾测试集预测正确数： ' num2str(number_B_sim)...
      '彩鸢尾测试集预测错误数： ' num2str(number_B - number_B_sim)...
      '彩鸢尾测试集预测正确率=' num2str(number_B_sim/number_B*100) '%']);
disp(['维吉尼亚鸢尾测试集预测正确数： ' num2str(number_C_sim)...
      '维吉尼亚鸢尾测试集预测错误数： ' num2str(number_C - number_C_sim)...
      '维吉尼亚鸢尾测试集预测正确率=' num2str(number_C_sim/number_C*100) '%']);
figure(2)
plot(1:10,T_test,'bo',1:10,T_sim_2,'r-*')
grid on
xlabel('测试集样本编号')
ylabel('测试集样本类别')
string = {'鸢尾花测试集预测结果(SSA-ELM)';['(正确率Accuracy = ' num2str(Accuracy_2) '%)' ]};
title(string)
legend('真实值','SSA-ELM预测值')
  
  
