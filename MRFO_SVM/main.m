tic % ��ʱ��
% ��ջ�����׼������
close all
clear;
clc;
format compact
addpath('addpath_svm')
%% ��ȡ����
data=xlsread('wine.xls');%3����
n=randperm(size(data,1));
data=data(n,:);
input=data(:,1:end-1);
output=data(:,end);
%% ѡ����Լ���ѵ����
p=round(size(data,1)*0.8);
train_wine=input(1:p,:);
test_wine=input(p+1:end,:);
train_wine_labels=output(1:p,:);
test_wine_labels=output(p+1:end,:);
label_test=test_wine_labels;

[mtrain,ntrain] = size(train_wine);%%�б�ʾ������ �б�ʾ����
[mtest,ntest] = size(test_wine);
dataset = [train_wine;test_wine];

[dataset_scale,ps] = mapminmax(dataset',0,1);
dataset_scale = dataset_scale';
train_wine = dataset_scale(1:mtrain,:);
test_wine = dataset_scale( (mtrain+1):(mtrain+mtest),: );
%% %%%%%%%%%%%%%%�Ż�SVM�еĲ���c��g��ʼ%%%%%%%%%%%%%%%%%%%%
Max_iter=50; % ����������
N=30;
% ���Ż�������Ϣ
dim=2; % ���Ż���������������Ϊc��g����
ub=[300 10]; % ����ȡֵ�Ͻ磬�˴���c��g���Ͻ���Ϊ500  100
lb=[0.0001 0.0001]; % ����ȡֵ�½磬�˴���c��g���½���Ϊ0.0001
v=5; %������֤
fobj=@(x)fun(x,train_wine_labels,train_wine,v);
%% ��������
Tn_sim=svm(train_wine,train_wine_labels,test_wine,test_wine_labels);
[best_p,fbest,fbest_store ]=MRFO(N,Max_iter,lb,ub,dim,fobj);         %

%% Ѱ�Ž��չʾ
bestc=best_p(1);
bestg=best_p(2);
figure
plot(fbest_store,'linewidth',1.5);
grid on;
xlabel('��������')
ylabel('��Ӧ��ֵ')
title('��������')

%% ��ӡ����ѡ��������������������һ�������㷨Ѱ�ŵõ��Ĳ���

disp('��ӡѡ����');
str=sprintf('Best c = %g��Best g = %g',bestc,bestg);
disp(str)

%% ������ѵĲ�������SVM����ѵ��
cmd_svm = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
model_svm = svmtrain(train_wine_labels,train_wine,cmd_svm);

%% SVM����Ԥ��
[predict_train,accuracy_train,decision_train] = svmpredict(train_wine_labels,train_wine,model_svm);
[predict_label,accuracy,decision_values] = svmpredict(test_wine_labels,test_wine,model_svm);

[labels_train1,index]=sort(train_wine_labels);
sim_train1=predict_train(index);
figure;
hold on;
plot(labels_train1,'bo');
plot(sim_train1,'r*');
xlabel('���Լ�����','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ��ѵ��������','Ԥ��ѵ��������');
title('ѵ��������ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on
snapnow

[labels_test1,index]=sort(test_wine_labels);
sim_test1=predict_label(index);

figure;
hold on;
plot(labels_test1,'Bo');
plot(sim_test1,'R*');
xlabel('���Լ�����','FontSize',12);
ylabel('����ǩ','FontSize',12);
legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
title('���Լ���ʵ�ʷ����Ԥ�����ͼ','FontSize',12);
grid on
snapnow


%% ��������

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
%% ������ͼ
[Metrics]=polygonareametric(test_wine_labels, predict_label);
%% ROC����
[TPR_rf,FPR_rf,TRF_rf,AUCRF_rf]=perfcurve(test_wine_labels, predict_label,1);
figure
plot(FPR_rf,TPR_rf,'r--','linewidth',1.5)
xlabel('��������FPR');ylabel('��������TPR')
hold on 
line([0 1],[0,1],'linewidth',1,'color','b'); 
hold on 
axis([0 1 0 1]); 
AUC_rf =trapz(FPR_rf,TPR_rf); 
string={'ROC���߽��'}; 
title(string) 
hold on 
text1=strcat('���ɭ�� AUC=',num2str(AUC_rf)); 
legend(text1);

