function [oa, aa, K, ua]=confusion(true_label,estim_label)

l=length(true_label);
nb_c=max(true_label);
confu=zeros(nb_c,nb_c);

for i=1:l
  confu(estim_label(i),true_label(i))= confu(estim_label(i),true_label(i))+1;
end

oa=trace(confu)/sum(confu(:)); %overall accurac
ua=diag(confu)./sum(confu,2);  %class accuracy
ua(isnan(ua))=0;
number=size(ua,1);

aa=sum(ua)/number;

Po=oa;
Pe=(sum(confu)*sum(confu,2))/(sum(confu(:))^2);

K=(Po-Pe)/(1-Pe);
%%
disp({'总体精度OA为：',num2str(oa)});
disp({'kappa系数为：',num2str(K)});
disp({'平均类别精度为：',num2str(aa)});
% disp({'类别精度UA为：',num2str(ua)});
