%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  读取数据
res = xlsread('数据集.xlsx');

%%  分析数据
num_class = length(unique(res(:, end)));  % 类别数（Excel最后一列放类别）
num_res = size(res, 1);                   % 样本数（每一行，是一个样本）
num_size = 0.7;                           % 训练集占数据集的比例
res = res(randperm(num_res), :);          % 打乱数据集（不打乱数据时，注释该行）

%%  设置变量存储数据
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 循环取出不同类别的样本
    mid_size = size(mid_res, 1);                    % 得到不同类别样本个数
    mid_tiran = round(num_size * mid_size);         % 得到该类别的训练样本个数

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 训练集输入
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % 训练集输出

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 测试集输入
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 测试集输出
end

%%  数据转置
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  得到训练集和测试样本个数
M = size(P_train, 2);
N = size(P_test , 2);
%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = T_train;
t_test  = T_test ;

%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';
%%  参数设置
pop = 10;                 % 种群数目
Max_iter =50;           % 迭代次数
dim = 2;                 % 优化参数个数
lb = [0.01, 0.01];       % 参数下限
ub = [ 100,  100];       % 参数上限

%%  优化函数
fobj = @(x)fitnessfun(x, p_train, t_train);

%%  优化算法
[ Best_score,Best_pos, curve] = COA(pop, Max_iter, lb, ub, dim, fobj);

%%  获取最佳参数
bestc = Best_pos(1);  
bestg = Best_pos(2);

%%  模型训练
cmd = [' -t 2 ', ' -c ', num2str(bestc), ' -g ', num2str(bestg)];
model = svmtrain(t_train, p_train, cmd);

%%  仿真测试
[T_sim1, error_1] = svmpredict(t_train, p_train, model);
[T_sim2, error_2] = svmpredict(t_test , p_test , model);

%%  性能评价
error1 = sum((T_sim1' == T_train))/M * 100 ;
error2 = sum((T_sim2' == T_test)) /N * 100 ;

%%  绘图
figure
plot(1: M, T_train, 'r*', 1: M, T_sim1, 'bo', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r*', 1: N, T_sim2, 'bo', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
grid

%%  适应度曲线
figure;
plot(1 : length(curve), curve, 'LineWidth', 1.5);
title('适应度曲线', 'FontSize', 13);
xlabel('迭代次数', 'FontSize', 10);
ylabel('适应度值', 'FontSize', 10);
xlim([1, length(curve)])
grid

