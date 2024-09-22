function   [A,P,S,F1]=calculate_err(Tn_sim,label_test)
class=unique(label_test); %class ��ʾ�������
for i=1:length(class) %������Ϊ1��������Ϊ0
    [~,num]=find(label_test==class(i));  %��ѭ��1Ϊ������ǩ1��Ϊ��������ǩ2/3��Ϊ����
    temp1=zeros(1,length(label_test));
    temp1(num)=1;
    
    [~,num2]=find(Tn_sim==class(i));  %��ѭ��1Ϊ������ǩ1��Ϊ��������ǩ2/3��Ϊ����
    temp2=zeros(1,length(label_test));
    temp2(num2)=1;
    %%%%%%%%%%%%%% ����Ϊ�����ӷ����� %%%%%%%%%%%%   
    TP=sum(label_test(num)==Tn_sim(num));  %���Ϊ1��������ϵͳ��ȷ�ж�Ϊ���1
    FN=sum(label_test(num)~=Tn_sim(num));  %���Ϊ1��������ϵͳ��ȷ�ж�Ϊ���0
    num_f=setdiff(1:length(label_test),num);  %���Ϊ0������
    TN=sum(label_test(num_f)==Tn_sim(num_f));  %���Ϊ0��������ϵͳ��ȷ�ж�Ϊ���0
    FP=sum(label_test(num_f)~=Tn_sim(num_f));  %���Ϊ0��������ϵͳ��ȷ�ж�Ϊ���0
    a(i)=(TP + TN)/(TP + FN + FP + TN); %��ӳ�˷�����������ϵͳ���ж�����
    p(i)=TP/(TP+FP);
    r(i)=TP/(TP+FN);
    s(i)=TN/(TN+FP);
    f1(i)=(1/p(i)+1/r(i))*0.5;
end
A=mean(a);P=mean(p);
S=mean(s);F1=mean(f1);