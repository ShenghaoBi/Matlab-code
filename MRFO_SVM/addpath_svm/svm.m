function [predict_label]=svm(train_wine,train_wine_labels,test_wine,test_wine_labels)
model=libsvmtrain(train_wine_labels,train_wine);
[predict_label, accuracy, dec_values]=libsvmpredict(test_wine_labels,test_wine,model);
acc=accuracy(1)/100;
%% �������
% ���Լ���ʵ�ʷ����Ԥ�����ͼ
% figure;
% hold on;
% plot(test_wine_labels,'o');
% plot(predict_label,'r*');
% xlabel('���Լ�����','FontSize',12);
% ylabel('����ǩ','FontSize',12);
% legend('ʵ�ʲ��Լ�����','Ԥ����Լ�����');
% title('SVM���Լ�������','FontSize',12);
% grid on
end