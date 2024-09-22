%% 蝠鲼觅食优化算法
function [BestX,BestF,HisBestFit]=MRFO(nPop,MaxIt,Low,Up,Dim,fobj)

    %初始化种群并获取适应度值
    for i=1:nPop   
        PopPos(i,:)=rand(1,Dim).*(Up-Low)+Low;
        PopFit(i)=fobj(PopPos(i,:));   
    end
       BestF=inf;
       BestX=[];
    %记录初始最优适应度值
    for i=1:nPop
        if PopFit(i)<=BestF
           BestF=PopFit(i);
           BestX=PopPos(i,:);
        end
    end

    HisBestFit=zeros(MaxIt,1);


for It=1:MaxIt  
     Coef=It/MaxIt; 
     
        % 对最优位置判断进行何种操作
       if rand<0.5 %进行螺旋觅食操作
          r1=rand;                         
          Beta=2*exp(r1*((MaxIt-It+1)/MaxIt))*(sin(2*pi*r1));    
          if  Coef>rand                                                      
              newPopPos(1,:)=BestX+rand(1,Dim).*(BestX-PopPos(1,:))+Beta*(BestX-PopPos(1,:)); %Equation (4)
          else
              IndivRand=rand(1,Dim).*(Up-Low)+Low;                                
              newPopPos(1,:)=IndivRand+rand(1,Dim).*(IndivRand-PopPos(1,:))+Beta*(IndivRand-PopPos(1,:)); %Equation (7)         
          end              
       else %链式觅食操作
            Alpha=2*rand(1,Dim).*(-log(rand(1,Dim))).^0.5;           
            newPopPos(1,:)=PopPos(1,:)+rand(1,Dim).*(BestX-PopPos(1,:))+Alpha.*(BestX-PopPos(1,:)); %Equation (1)
       end
      % 对其他位置判断进行何种操作
    for i=2:nPop
        if rand<0.5%进行螺旋觅食操作
           r1=rand;                         
           Beta=2*exp(r1*((MaxIt-It+1)/MaxIt))*(sin(2*pi*r1));    
             if  Coef>rand                                                      
                 newPopPos(i,:)=BestX+rand(1,Dim).*(PopPos(i-1,:)-PopPos(i,:))+Beta*(BestX-PopPos(i,:)); %Equation (4)
             else
                 IndivRand=rand(1,Dim).*(Up-Low)+Low;                                
                 newPopPos(i,:)=IndivRand+rand(1,Dim).*(PopPos(i-1,:)-PopPos(i,:))+Beta*(IndivRand-PopPos(i,:));  %Equation (7)       
             end              
        else %链式觅食操作
            Alpha=2*rand(1,Dim).*(-log(rand(1,Dim))).^0.5;           
            newPopPos(i,:)=PopPos(i,:)+rand(1,Dim).*(PopPos(i-1,:)-PopPos(i,:))+Alpha.*(BestX-PopPos(i,:)); %Equation (1)
       end         
    end
    %计算更新后的适应度值     
   for i=1:nPop        
       newPopPos(i,:)=SpaceBound(newPopPos(i,:),Up,Low);
       newPopFit(i)=fobj(newPopPos(i,:));    
      if newPopFit(i)<PopFit(i)
         PopFit(i)=newPopFit(i);
         PopPos(i,:)=newPopPos(i,:);
      end
   end
    %种群进行翻滚觅食操作
    S=2;
    for i=1:nPop           
        newPopPos(i,:)=PopPos(i,:)+S*(rand*BestX-rand*PopPos(i,:)); %Equation (8)
    end
     %更新种群，用适应度较好的位置代替适应度较差的位置。
     for i=1:nPop        
         newPopPos(i,:)=SpaceBound(newPopPos(i,:),Up,Low);
         newPopFit(i)=fobj(newPopPos(i,:));    
         if newPopFit(i)<PopFit(i)
            PopFit(i)=newPopFit(i);
            PopPos(i,:)=newPopPos(i,:);
         end
     end
     %更新全局最优位置
     for i=1:nPop
        if PopFit(i)<BestF
           BestF=PopFit(i);
           BestX=PopPos(i,:);            
        end
     end
 
  
     HisBestFit(It)=BestF;
end


