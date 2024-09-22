function fitness=fun(x,train_label,train_in,v)
        cmd = ['-v ',num2str(v),' -c ',num2str(x(1)),' -g ',num2str( x(2) )];
        fitness = svmtrain(train_label, train_in, cmd);
        fitness = 100-fitness;
end