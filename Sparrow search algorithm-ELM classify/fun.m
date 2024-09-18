%适应度函数，以训练集和测试集的分类错误率作为适应度值
function [fitness,IW,B,LW,TF,TYPE] = fun(x,P_train,T_train,N,P_test,T_test)
R = size(P_train,1);
S = size(T_train,1);
IW = x(1:N*R);
B = x(N*R+1:N*R+N*S);
IW = reshape(IW,[N,R]);
B = reshape(B,N,1);
TYPE = 1;%回归
TF = 'sig';
[IW,B,LW,TF,TYPE] = elmtrainNew(P_train,T_train,N,TF,TYPE,IW,B);
%% ELM仿真测试
T_sim_1 = elmpredict(P_train,IW,B,LW,TF,TYPE);
T_sim_2 = elmpredict(P_test,IW,B,LW,TF,TYPE);
% 训练集正确率
k1 = length(find(T_train == T_sim_1));
n1 = length(T_train);
Accuracy_1 = k1 / n1 ;
% 测试集正确率
k2 = length(find(T_test == T_sim_2));
n2 = length(T_test);
Accuracy_2 = k2 / n2 ;
%错误率
fitness = 2 - Accuracy_1 - Accuracy_2;
end