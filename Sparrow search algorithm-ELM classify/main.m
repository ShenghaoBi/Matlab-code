%_________________________________________________________________________%
% ������ȸ�Ż�ELM�����������        %
%_________________________________________________________________________%
clear all 
clc
%% ��������
data=xlsread('tdata','all data','B2:E51');
Train = xlsread('tdata','training set','B2:E41');
Test = xlsread('tdata','test set','B2:E11');
% ѵ������
P_train = xlsread('tdata','training set','B2:D41')';
T_train = xlsread('tdata','training set','E2:E41')';
% ��������
P_test = xlsread('tdata','test set','B2:D11')';
T_test = xlsread('tdata','test set','E2:E11')';
%% ���϶���

%% ------����SSA��ELM���з���----------------------------------------- 
%ѵ��������سߴ�
R = size(P_train,1);
S = size(T_train,1);
N = 100;%���������
%% ������ȸ�Ż�����
pop=20; %��Ⱥ����
Max_iteration=200; %  �趨����������
dim = N*R + N;%ά�ȣ���Ȩֵ����ֵ�ĸ���
lb = [-1.*ones(1,N*R),zeros(1,N)];%�±߽�
ub = [ones(1,N*R),ones(1,N)];%�ϱ߽�
fobj = @(x) fun(x,P_train,T_train,N,P_test,T_test);
[Best_pos,Best_score,SSA_curve]=SSA(pop,Max_iteration,lb,ub,dim,fobj); %��ʼ�Ż�
[fitness,IW,B,LW,TF,TYPE] = fun(Best_pos,P_train,T_train,N,P_test,T_test);%��ȡ�Ż������ز���
figure
plot(SSA_curve,'linewidth',1.5);
grid on
xlabel('��������')
ylabel('��Ӧ�Ⱥ���')
title('SSA-ELM��������')
%% ELM�������
T_sim_1 = elmpredict(P_train,IW,B,LW,TF,TYPE);
T_sim_2 = elmpredict(P_test,IW,B,LW,TF,TYPE);

%% ����Ա�
disp('SSA-ELM���չʾ��----------------')
result_1 = [T_train' T_sim_1'];
result_2 = [T_test' T_sim_2'];
% ѵ������ȷ��
k1 = length(find(T_train == T_sim_1));
n1 = length(T_train);
Accuracy_1 = k1 / n1 * 100;
disp(['ѵ������ȷ��Accuracy = ' num2str(Accuracy_1) '%(' num2str(k1) '/' num2str(n1) ')'])
% ���Լ���ȷ��
k2 = length(find(T_test == T_sim_2));
n2 = length(T_test);
Accuracy_2 = k2 / n2 * 100;
disp(['���Լ���ȷ��Accuracy = ' num2str(Accuracy_2) '%(' num2str(k2) '/' num2str(n2) ')'])
%% ��ʾ
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
disp(['�β��������' num2str(50)...
      '  ɽ�β��' num2str(total_A)...
      '  ���β��' num2str(total_B)...
      '  ά�������β��' num2str(total_C)]);
disp(['ѵ�����β��������' num2str(40)...
      '  ɽ�β��' num2str(count_A)...
      '  ���β��' num2str(count_B)...
      '  ά�������β��' num2str(count_C)]);
disp(['���Լ��β��������' num2str(10)...
      '  ɽ�β��' num2str(number_A)...
      '  ���β��' num2str(number_B)...
      '  ά�������β��' num2str(number_B)]);
disp(['ɽ�β���Լ�Ԥ����ȷ���� ' num2str(number_A_sim)...
      'ɽ�β���Լ�Ԥ��������� ' num2str(number_A - number_A_sim)...
      'ɽ�β���Լ�Ԥ����ȷ��=' num2str(number_A_sim/number_A*100) '%']);
disp(['���β���Լ�Ԥ����ȷ���� ' num2str(number_B_sim)...
      '���β���Լ�Ԥ��������� ' num2str(number_B - number_B_sim)...
      '���β���Լ�Ԥ����ȷ��=' num2str(number_B_sim/number_B*100) '%']);
disp(['ά�������β���Լ�Ԥ����ȷ���� ' num2str(number_C_sim)...
      'ά�������β���Լ�Ԥ��������� ' num2str(number_C - number_C_sim)...
      'ά�������β���Լ�Ԥ����ȷ��=' num2str(number_C_sim/number_C*100) '%']);
figure(2)
plot(1:10,T_test,'bo',1:10,T_sim_2,'r-*')
grid on
xlabel('���Լ��������')
ylabel('���Լ��������')
string = {'�β�����Լ�Ԥ����(SSA-ELM)';['(��ȷ��Accuracy = ' num2str(Accuracy_2) '%)' ]};
title(string)
legend('��ʵֵ','SSA-ELMԤ��ֵ')
  
  
