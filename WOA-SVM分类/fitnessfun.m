function fitness = fitnessfun(x, p_train, t_train)

%%  获取最优参数
c = x(1);
g = x(2);

%%  模型训练
cmd = [' -v 3 ', ' -c ',num2str(c), ' -g ', num2str(g)];
accuracy = svmtrain(t_train, p_train, cmd); 

%%  以分类预测错误率作为优化的目标函数值
if size(accuracy, 1) == 0
    fitness = 100;
else
    fitness = (100 - accuracy(1)) / 100;
end

end