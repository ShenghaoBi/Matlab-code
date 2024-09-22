function [predict_label]=svm(train_wine,train_wine_labels,test_wine,test_wine_labels)
model=libsvmtrain(train_wine_labels,train_wine);
[predict_label, accuracy, dec_values]=libsvmpredict(test_wine_labels,test_wine,model);
acc=accuracy(1)/100;
%% 结果分析
% 测试集的实际分类和预测分类图
% figure;
% hold on;
% plot(test_wine_labels,'o');
% plot(predict_label,'r*');
% xlabel('测试集样本','FontSize',12);
% ylabel('类别标签','FontSize',12);
% legend('实际测试集分类','预测测试集分类');
% title('SVM测试集分类结果','FontSize',12);
% grid on
end