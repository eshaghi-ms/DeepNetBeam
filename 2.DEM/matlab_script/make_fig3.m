function make_fig3(Data,L , i,j,k,l, VarList, titles, limits, lineStyles)

numElemU = 20;
numElemV = 10;
numGauss = 4;
numPtsU = 2*numGauss*numElemU;
numPtsV = 2*numGauss*numElemV;

H = 0.1;

P = length(i);
Q = length(j);
R = length(k);
S = length(l);

VarLen = length(VarList);

sample(P,Q,R,S).X = [];
sample(P,Q,R,S).X = [];


for p=1:P
    for q=1:Q
        for r=1:R
            for s=1:S
                sample(p,q,r,s).X = Data(i(p),j(q),k(r),l(s)).xPhys;
                sample(p,q,r,s).Y = Data(i(p),j(q),k(r),l(s)).yPhys;
                sample(p,q,r,s).X = reshape(sample(p,q,r,s).X,numPtsU,numPtsV);
                sample(p,q,r,s).Y = reshape(sample(p,q,r,s).Y,numPtsU,numPtsV);
                for row=1:VarLen
                    sample(p,q,r,s).(VarList(row)) = ...
                                Data(i(p),j(q),k(r),l(s)).(VarList(row));
                    sample(p,q,r,s).(VarList(row)) = ...
                                reshape(sample(p,q,r,s).(VarList(row)),numPtsU,numPtsV);
                end
            end
        end
    end
end


PL = 1; %PlotLength
COL = 3;
ROW = 6;


z = sample(1,1,1,1).Y(numPtsU/2,:)/H-0.5;
%sigma = zeros(size(z));

x_l = [0.1, 0.5, 0.9];

LegendsBot = {'$e_{0} = 0.0$','$e_{0} = 0.1$','$e_{0} = 0.2$','$e_{0} = 0.3$','$e_{0} = 0.4$',...
              '$e_{0} = 0.5$','$e_{0} = 0.6$','$e_{0} = 0.7$','$e_{0} = 0.8$'};

LegendsTop = {'$L/H = 1$','$L/H = 2$','$L/H = 3$','$L/H = 5$','$L/H = 10$',...
              '$L/H = 20$','$L/H = 30$','$L/H = 40$','$L/H = 50$'};

for row=5:6
    for col = 1:COL
        for r=1:R
            sigma = sample(row-4,1,r,6).stress_xx(x_l(col)*numPtsU,:); % (q, s) is assumed equal (1, 6)
            sigma = sigma / max(abs(sigma));
            subplot(ROW,COL*PL, [PL*((row-1)*COL+col-1)+1  PL*((row-1)*COL+col-1)+PL]);

            plot(z, sigma, lineStyles{r},'LineWidth',0.5)
            hold on
        end

        if row==6
            xlabel('$(z/H)$', 'Interpreter', 'latex')
        end
        if col==1 && row==6
            ylabel("($$\sigma_{xx})$$, Normalized", 'Interpreter', 'latex', 'Position',[-0.6,1.4,0])
        end
        xticks(-0.5:0.25:.5);
        yticks(-1:0.5:1);
        yticklabels({'-1.0','-0.5','0.0','0.5','1.0',})
        grid on
        
        if row==6 && col==COL
            legend(LegendsBot, 'Position', [0.95, 0.22 , 0, 0], 'Interpreter', 'latex', ...
                         'Box', 'off','FontSize',12)
        end
        
        % Changing font size for axes numbers (tick labels)
        set(gca, 'FontSize', 9, 'FontName', 'Times New Roman') 
        % Changing font size for axes titles
        set(get(gca, 'XLabel'), 'FontSize', 10)  % Set font size to 14 for x-axis label
        set(get(gca, 'YLabel'), 'FontSize', 10)  % Set font size to 14 for y-axis label
        set(get(gca, 'Title'), 'FontSize', 14)    % Set font size to 16 for title
        set(get(gca, 'legend'), 'FontSize', 10)    % Set font size to 16 for title

        hold off

    end
end

for row=1:2
    for col = 1:COL
        for s=1:S
            sigma = sample(row,1,6,s).stress_xx(x_l(col)*numPtsU,:); % (q, r) is assumed equal (1, 6)
            subplot(ROW,COL*PL, [PL*((row-1)*COL+col-1)+1  PL*((row-1)*COL+col-1)+PL]);

            plot(z, sigma, lineStyles{s},'LineWidth',0.5)
            hold on
        end
        if row==2
            xlabel('$(z/H)$', 'Interpreter', 'latex')
        end

        if col==1 && row==1
            ylabel("($$\sigma_{xx})$$, Normalized", 'Interpreter', 'latex', 'Position',[-0.6,-0.99,0])
        end

        xticks(-0.5:0.25:.5);
        %yticks(-1:0.5:1);
        %yticklabels({'-1.0','-0.5','0.0','0.5','1.0',})
        grid on
        
        if row==2 && col==COL
            legend(LegendsTop, 'Position', [0.95, 0.8 , 0, 0], 'Interpreter', 'latex', ...
                         'Box', 'off','FontSize',12)
        end
        
        % Changing font size for axes numbers (tick labels)
        set(gca, 'FontSize', 9, 'FontName', 'Times New Roman') 
        % Changing font size for axes titles
        set(get(gca, 'XLabel'), 'FontSize', 10)  % Set font size to 14 for x-axis label
        set(get(gca, 'YLabel'), 'FontSize', 10)  % Set font size to 14 for y-axis label
        set(get(gca, 'Title'), 'FontSize', 14)    % Set font size to 16 for title
        set(get(gca, 'legend'), 'FontSize', 10)    % Set font size to 16 for title

        hold off
    end
end



%subplot(ROW,COL*PL, [PL*((3-1)*COL+1-1)+1+PL  PL*((3-1)*COL+3-1)]);


axes('Position',[0.225 0.46 0.55 0.08],'Box','on');
contourf(sample(1,1,6,6).X/max(sample(1,1,6,6).X,[],'all'), sample(1,1,6,6).Y, ...
         sample(1,1,6,6).(VarList(1)),500,'edgecolor','none')
%colorbar( target , 'off' ) 
xticks(-0:0.1:1);
yticklabels({'-0.5','0.0','0.5'})
xlabel('$(x/L)$', 'Interpreter', 'latex')
ylabel('$(z)$', 'Interpreter', 'latex')
title( "($$\;\sigma_{xx}\;)$$ H-C Beam" , 'Interpreter','latex', 'Position',[1.15, 0.033, 0] )


% Changing font size for axes numbers (tick labels)
set(gca, 'FontSize', 9, 'FontName', 'Times New Roman') 
% Changing font size for axes titles
set(get(gca, 'XLabel'), 'FontSize', 10)  % Set font size to 14 for x-axis label
set(get(gca, 'YLabel'), 'FontSize', 10)  % Set font size to 14 for y-axis label
set(get(gca, 'Title'), 'FontSize', 14)    % Set font size to 16 for title
set(get(gca, 'legend'), 'FontSize', 10)    % Set font size to 16 for title

%set(gca,'Xticklabel',[])
%set(gca,'Yticklabel',[])
colormap jet
clim([-10, 10]);

for i = x_l
    x_value = i;
    y_range = ylim;  
    hold on; 
    plot([x_value, x_value], y_range, 'r--', 'LineWidth', 2);
end

annotation('line',[0.225+0.55*x_l(1), 0.13],[0.46, 0.355],"LineStyle",":","Color","r","LineWidth",1.2)
annotation('line',[0.225+0.55*x_l(1), 0.343],[0.46, 0.355],"LineStyle",":","Color","r","LineWidth",1.2)

difx = 0.28;
annotation('line',[0.225+0.55*x_l(2), 0.13+difx],[0.46, 0.355],"LineStyle",":","Color","r","LineWidth",1.2)
annotation('line',[0.225+0.55*x_l(2), 0.343+difx],[0.46, 0.355],"LineStyle",":","Color","r","LineWidth",1.2)

difx = 0.561;
annotation('line',[0.225+0.55*x_l(3), 0.13+difx],[0.46, 0.355],"LineStyle",":","Color","r","LineWidth",1.2)
annotation('line',[0.225+0.55*x_l(3), 0.343+difx],[0.46, 0.355],"LineStyle",":","Color","r","LineWidth",1.2)

dify = 0.325;
annotation('line',[0.225+0.55*x_l(1), 0.13 ],[0.46+0.08, 0.355+dify],"LineStyle",":","Color","r","LineWidth",1.2)
annotation('line',[0.225+0.55*x_l(1), 0.343],[0.46+0.08, 0.355+dify],"LineStyle",":","Color","r","LineWidth",1.2)

difx = 0.28;
annotation('line',[0.225+0.55*x_l(2), 0.13+difx],[0.46+0.08, 0.355+dify],"LineStyle",":","Color","r","LineWidth",1.2)
annotation('line',[0.225+0.55*x_l(2), 0.343+difx],[0.46+0.08, 0.355+dify],"LineStyle",":","Color","r","LineWidth",1.2)

difx = 0.561;
annotation('line',[0.225+0.55*x_l(3), 0.13+difx],[0.46+0.08, 0.355+dify],"LineStyle",":","Color","r","LineWidth",1.2)
annotation('line',[0.225+0.55*x_l(3), 0.343+difx],[0.46+0.08, 0.355+dify],"LineStyle",":","Color","r","LineWidth",1.2)


text(0.035     ,  0.17,"($$\;x/L\; = \;" + num2str(x_l(1),'%.1f') + ")$$ " , 'Interpreter','latex')
text(0.035     , -0.07,"($$\;x/L\; = \;" + num2str(x_l(1),'%.1f') + ")$$ " , 'Interpreter','latex')
text(0.035+0.42,  0.17,"($$\;x/L\; = \;" + num2str(x_l(2),'%.1f') + ")$$ " , 'Interpreter','latex')
text(0.035+0.42, -0.07,"($$\;x/L\; = \;" + num2str(x_l(2),'%.1f') + ")$$ " , 'Interpreter','latex')
text(0.035+0.855,  0.17,"($$\;x/L\; = \;" + num2str(x_l(3),'%.1f') + ")$$ " , 'Interpreter','latex')
text(0.035+0.855, -0.07,"($$\;x/L\; = \;" + num2str(x_l(3),'%.1f') + ")$$ " , 'Interpreter','latex')
hold off


















end