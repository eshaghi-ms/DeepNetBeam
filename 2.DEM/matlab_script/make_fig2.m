function make_fig2(Data,L , i,j,k,l, VarList, titles, limits, lineStyles,Legends)

numElemU = 20;
numElemV = 10;
numGauss = 4;
numPtsU = 2*numGauss*numElemU;
numPtsV = 2*numGauss*numElemV;

%L = [0.1, 0.2, 0.3 ,0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
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
                    sample(p,q,r,s).maxW = min(sample(p,q,r,s).(VarList(row)), [], "all");
                    sample(p,q,r,s).(VarList(row)) = ...
                                reshape(sample(p,q,r,s).(VarList(row)),numPtsU,numPtsV);
                end
            end
        end
    end
end


%aspect = [10 1 1];
%FontS = 10;
%TextSize = 14;


PL = 2; %PlotLength

COL = Q;
ROW = P;
y(S)=0;
x = L/H;
xx = linspace(min(x), max(x), 100);

aspect = [10 1 1];
FontS = 14;
TextSize = 12;

% Degree of the polynomial (you can adjust this)
degree = 3;

%% main Figure


for var=1:ROW
    for col=1:COL
        for r=1:R
            for s=1:S
                y(s) = sample(var,col,r,s).maxW;
            end


            % Fit a polynomial to the data
            p = polyfit(x, y, degree);
            yy = polyval(p, xx);
            
            subplot(ROW,COL*PL, [PL*((var-1)*COL+col-1)+1  PL*((var-1)*COL+col-1)+PL]);

            % Create a plot with smooth line
            plot(xx, yy, lineStyles{r})
            hold on 
  
        end

        % Adding labels and title for clarity
        xlabel('Slenderness ratio $(L/H)$', 'Interpreter', 'latex')
        if col == 1
            ylabel('Maximum deflection', 'Interpreter', 'latex')
        end
        title(titles(var,col),'Interpreter','latex')

        ylim(limits{var,col})

        grid on
        legend(Legends, 'Location', 'southwest', 'Interpreter', 'latex', ...
                            'Box', 'off','FontSize',TextSize, 'NumColumns', 2)
        % Changing font size for axes numbers (tick labels)
        set(gca, 'FontSize', 13, 'FontName', 'Times New Roman') 
        % Changing font size for axes titles
        set(get(gca, 'XLabel'), 'FontSize', 12)  % Set font size to 14 for x-axis label
        set(get(gca, 'YLabel'), 'FontSize', 12)  % Set font size to 14 for y-axis label
        set(get(gca, 'Title'), 'FontSize', 14)    % Set font size to 16 for title
        set(get(gca, 'legend'), 'FontSize', 12)    % Set font size to 16 for title



        hold off
        clear p
        clear yy
    end
end


%% magnified Figures
magFx = 2.75;
magFy = 3.5;

StartPx = 0.1315;
StartPy = 0.425;

difx = 0.0115;
dify = -0.1850;

difcol1 = .2686;
difcol2 = .538;
difrow = 0.472;

xs1 = [ StartPx , StartPx+difcol1 , StartPx+difcol2 ;
        StartPx , StartPx+difcol1 , StartPx+difcol2 ];

xe1 = xs1+difx;
ys1 = [ StartPy+difrow , StartPy+difrow , StartPy+difrow ;
        StartPy        , StartPy        , StartPy     ];

ye1 = ys1+dify;

ls = 0.0474;
hs = 0.03;
le = magFx*ls;
he = magFy*hs;

Ylim{1,1} = [-4e-5,0];
Ylim{1,2} = [-6e-5,0];
Ylim{1,3} = [-10e-5,0];
Ylim{2,1} = [-4e-5,0];
Ylim{2,2} = [-6e-5,0];
Ylim{2,3} = [-10e-5,0];

for var = 1:2
    for col = 1:3
        magnify([xs1(var,col), xe1(var,col), ys1(var,col), ye1(var,col)], ...
                [ls, hs, le, he],...
                 x, sample, degree, lineStyles, R, var, col,Ylim{var,col})
    end
end

end

