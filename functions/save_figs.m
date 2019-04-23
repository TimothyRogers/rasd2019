function [] = save_figs(folder,fname)


savefig(fullfile(folder,fname));
% export_fig(fullfile(folder,fname),'-eps','-png','-transparent')
print(fullfile(folder,fname),'-depsc2')
print(fullfile(folder,fname),'-dpng','-r300')

end