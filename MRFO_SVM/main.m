tic % 计时器
% 清空环境，准备数据
close all
clear;
clc;
format compact
addpath('addpath_svm')
%% 读取数据
data=xlsread('wine.xls');%3分类
n=randperm(size(data,1));
data=data(n,:);
input=data(:,1:end-1);
output=data(:,end);
%% 选择测试集与训练集
p=round(size(data,1)*0.8);
train_wine=input(1:p,:);
test_wine=input(p+1:end,:);
train_wine_labels=output(1:p,:);
test_wine_labels=output(p+1:end,:);
label_test=test_wine_labels;

[mtrain,ntrain] = size(train_wine);%%行表示样本数 列表示特征
[mtest,ntest] = size(test_wine);
dataset = [train_wine;test_wine];

[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';
train_wine = dataset_scale(1:mtrain,:);
test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% %%%%%%%%%%%%%%优化SVM中的参数c和g开始%%%%%%%%%%%%%%%%%%%%
Max_iter=50; % 最大迭代次数
N=30;
% 待优化参数信息
dim=2; % 待优化参数个数，次数为c和g两个
ub=[300 10]; % 参数取值上界，此处将c和g的上界设为500  100
lb=[0.0001 0.0001]; % 参数取值下界，此处将c和g的下界设为0.0001
v=5; %交叉验证
fobj=@(x)fun(x,train_wine_labels,train_wine,v);
%% 迭代计算
Tn_sim=svm(train_wine,train_wine_labels,test_wine,test_wine_labels);
[best_p,fbest,fbest_store ]=MRFO(N,Max_iter,lb,ub,dim,fobj);         %

%% 寻优结果展示
bestc=best_p(1);
bestg=best_p(2);
figure
plot(fbest_store,'linewidth',1.5);
grid on;
xlabel('迭代次数')
ylabel('适应度值')
title('收敛曲线')

%% 打印参数选择结果，这里输出的是最后一次尊海鞘算法寻优得到的参数

disp('打印选择结果');
str=sprintf('Best c = %g，Best g = %g',bestc,bestg);
disp(str)

%% 利用最佳的参数进行SVM网络训练
cmd_svm = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model_svm = svmtrain(train_wine_labels,train_wine,cmd_svm);

%% SVM网络预测
[predict_train,accuracy_train,decision_train] = svmpredict(train_wine_labels,train_wine,model_svm);
[predict_label,accuracy,decision_values] = svmpredict(test_wine_labels,test_wine,model_svm);

[labels_train1,index]=sort(train_wine_labels);
sim_train1=predict_train(index);
figure;
hold on;
plot(labels_train1,'bo');
plot(sim_train1,'r*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际训练集分类','预测训练集分类');
title('训练集集的实际分类和预测分类图','FontSize',12);
grid on
snapnow

[labels_test1,index]=sort(test_wine_labels);
sim_test1=predict_label(index);

figure;
hold on;
plot(labels_test1,'Bo');
plot(sim_test1,'R*');
xlabel('测试集样本','FontSize',12);
ylabel('类别标签','FontSize',12);
legend('实际测试集分类','预测测试集分类');
title('测试集的实际分类和预测分类图','FontSize',12);
grid on
snapnow


%% 混淆矩阵

[confusionMatrix, classOrder] = confusionmat...
    (train_wine_labels, predict_train);
figure
set(gcf, 'position', [300 150 600 400])
confusionchart(confusionMatrix, classOrder,...
    'Title', '\fontname{Times New Roman}Training Sample',...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

[confusionMatrix, classOrder] = confusionmat...
    (test_wine_labels, predict_label);
figure
set(gcf, 'position', [300 150 600 400])
confusionchart(confusionMatrix, classOrder,...
    'Title', '\fontname{Times New Roman}Testing Sample',...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
%% 六边形图
[Metrics]=polygonareametric(test_wine_labels, predict_label);
%% ROC曲线
[TPR_rf,FPR_rf,TRF_rf,AUCRF_rf]=perfcurve(test_wine_labels, predict_label,1);
figure
plot(FPR_rf,TPR_rf,'r--','linewidth',1.5)
xlabel('假正类率FPR');ylabel('真正类率TPR')
hold on 
line([0 1],[0,1],'linewidth',1,'color','b'); 
hold on 
axis([0 1 0 1]); 
AUC_rf =trapz(FPR_rf,TPR_rf); 
string={'ROC曲线结果'}; 
title(string) 
hold on 
text1=strcat('随机森林 AUC=',num2str(AUC_rf)); 
legend(text1);

