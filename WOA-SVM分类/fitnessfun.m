function fitness = fitnessfun(x, p_train, t_train)

%%  ��ȡ���Ų���
c = x(1);
g = x(2);

%%  ģ��ѵ��
cmd = [' -v 3 ', ' -c ',num2str(c), ' -g ', num2str(g)];
accuracy = svmtrain(t_train, p_train, cmd); 

%%  �Է���Ԥ���������Ϊ�Ż���Ŀ�꺯��ֵ
if size(accuracy, 1) == 0
    fitness = 100;
else
    fitness = (100 - accuracy(1)) / 100;
end

end