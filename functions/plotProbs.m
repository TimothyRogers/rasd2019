function [map] = plotProbs(samples,variable_name)

[map,cdf,vals] = mapFromCDF(samples);

%yyaxis right
histogram(samples,100,'Normalization','pdf','EdgeColor','None','FaceAlpha',0.2)
ylabel('PDF')

%yyaxis left
%plot(vals,cdf,'LineWidth',2)
%line([0 map],[0.5 0.5],'Color','r','LineStyle','--','LineWidth',2)
%line([map map],[0 1],'Color','r','LineStyle','--','LineWidth',2)
%line([mean(samples) mean(samples)],[0 1],'Color','m','LineWidth',2,'LineStyle','-.')
%ylabel('CDF')
xlabel(variable_name,'interpreter','latex')


end