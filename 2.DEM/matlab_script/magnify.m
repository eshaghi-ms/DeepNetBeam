function magnify(XY, LH, x, sample, degree, lineStyles, R, row, col,Ylim)

xs1 = XY(1);
xe1 = XY(2);
ys1 = XY(3);
ye1 = XY(4);

ls = LH(1); 
hs = LH(2);
le = LH(3);
he = LH(4);

xs2 = xs1+ls;
xe2 = xe1+le;
ys2 = ys1+hs;
ye2 = ye1+he;

annotation("rectangle",[xs1, ys1, ls, hs],"EdgeColor","red","LineStyle","-.");

annotation('line',[xs1, xe1],[ys1, ye1],"LineStyle",":","Color","r","LineWidth",1.1)
annotation('line',[xs2, xe2],[ys2, ye2],"LineStyle",":","Color","r","LineWidth",1.1)


x = x(1:5);
clear y
y(5)=0;

xx = linspace(min(x), 10, 100);
axes('Position',[xe1 ye1 le he],'Box','on');


for r=1:R
    for s=1:5
        y(s) = sample(row,col,r,s).maxW;
    end
    
    % Fit a polynomial to the data
    p = polyfit(x, y, degree);
    yy = polyval(p, xx);
    
    % Create a plot with smooth line
    plot(xx, yy, lineStyles{r})
    hold on

end

ylim(Ylim)
xlim([1,10])

hold off
clear p
clear yy

end