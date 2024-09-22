%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������

%%  ��ȡ����
res = xlsread('���ݼ�.xlsx');

%%  ��������
num_class = length(unique(res(:, end)));  % �������Excel���һ�з����
num_res = size(res, 1);                   % ��������ÿһ�У���һ��������
num_size = 0.7;                           % ѵ����ռ���ݼ��ı���
res = res(randperm(num_res), :);          % �������ݼ�������������ʱ��ע�͸��У�

%%  ���ñ����洢����
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  �������ݼ�
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % ѭ��ȡ����ͬ��������
    mid_size = size(mid_res, 1);                    % �õ���ͬ�����������
    mid_tiran = round(num_size * mid_size);         % �õ�������ѵ����������

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % ѵ��������
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % ѵ�������

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % ���Լ�����
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % ���Լ����
end

%%  ����ת��
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  �õ�ѵ�����Ͳ�����������
M = size(P_train, 2);
N = size(P_test , 2);
%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = T_train;
t_test  = T_test ;

%%  ת������Ӧģ��
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';
%%  ��������
pop = 10;                 % ��Ⱥ��Ŀ
Max_iter =50;           % ��������
dim = 2;                 % �Ż���������
lb = [0.01, 0.01];       % ��������
ub = [ 100,  100];       % ��������

%%  �Ż�����
fobj = @(x)fitnessfun(x, p_train, t_train);

%%  �Ż��㷨
[ Best_score,Best_pos, curve] = COA(pop, Max_iter, lb, ub, dim, fobj);

%%  ��ȡ��Ѳ���
bestc = Best_pos(1);  
bestg = Best_pos(2);

%%  ģ��ѵ��
cmd = [' -t 2 ', ' -c ', num2str(bestc), ' -g ', num2str(bestg)];
model = svmtrain(t_train, p_train, cmd);

%%  �������
[T_sim1, error_1] = svmpredict(t_train, p_train, model);
[T_sim2, error_2] = svmpredict(t_test , p_test , model);

%%  ��������
error1 = sum((T_sim1' == T_train))/M * 100 ;
error2 = sum((T_sim2' == T_test)) /N * 100 ;

%%  ��ͼ
figure
plot(1: M, T_train, 'r*', 1: M, T_sim1, 'bo', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'ѵ����Ԥ�����Ա�'; ['׼ȷ��=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r*', 1: N, T_sim2, 'bo', 'LineWidth', 1)
legend('��ʵֵ', 'Ԥ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string = {'���Լ�Ԥ�����Ա�'; ['׼ȷ��=' num2str(error2) '%']};
title(string)
grid

%%  ��Ӧ������
figure;
plot(1 : length(curve), curve, 'LineWidth', 1.5);
title('��Ӧ������', 'FontSize', 13);
xlabel('��������', 'FontSize', 10);
ylabel('��Ӧ��ֵ', 'FontSize', 10);
xlim([1, length(curve)])
grid

