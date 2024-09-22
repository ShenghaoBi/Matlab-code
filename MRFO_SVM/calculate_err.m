function   [A,P,S,F1]=calculate_err(Tn_sim,label_test)
class=unique(label_test); %class 显示的类别数
for i=1:length(class) %正样本为1，负样本为0
    [~,num]=find(label_test==class(i));  %以循环1为例，标签1作为正例，标签2/3则为负例
    temp1=zeros(1,length(label_test));
    temp1(num)=1;
    
    [~,num2]=find(Tn_sim==class(i));  %以循环1为例，标签1作为正例，标签2/3则为负例
    temp2=zeros(1,length(label_test));
    temp2(num2)=1;
    %%%%%%%%%%%%%% 上述为建立子分类器 %%%%%%%%%%%%   
    TP=sum(label_test(num)==Tn_sim(num));  %类别为1的样本被系统正确判定为类别1
    FN=sum(label_test(num)~=Tn_sim(num));  %类别为1的样本被系统正确判定为类别0
    num_f=setdiff(1:length(label_test),num);  %类别为0的样本
    TN=sum(label_test(num_f)==Tn_sim(num_f));  %类别为0的样本被系统正确判定为类别0
    FP=sum(label_test(num_f)~=Tn_sim(num_f));  %类别为0的样本被系统正确判定为类别0
    a(i)=(TP + TN)/(TP + FN + FP + TN); %反映了分类器对整个系统的判定能力
    p(i)=TP/(TP+FP);
    r(i)=TP/(TP+FN);
    s(i)=TN/(TN+FP);
    f1(i)=(1/p(i)+1/r(i))*0.5;
end
A=mean(a);P=mean(p);
S=mean(s);F1=mean(f1);