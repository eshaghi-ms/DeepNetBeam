function make_fig1(Data, i,j,k,l, VarList, titles, limits)

numElemU = 20;
numElemV = 10;
numGauss = 4;
numPtsU = 2*numGauss*numElemU;
numPtsV = 2*numGauss*numElemV;

row = length(VarList);
col = length(k);

sample(col).X = [];
sample(col).X = [];

for p=1:col
    sample(p).X = Data(i,j,k(p),l).xPhys;
    sample(p).Y = Data(i,j,k(p),l).yPhys;
    sample(p).X = reshape(sample(p).X,numPtsU,numPtsV);
    sample(p).Y = reshape(sample(p).Y,numPtsU,numPtsV);
    for q=1:row
        sample(p).(VarList(q)) = Data(i,j,k(p),l).(VarList(q));
        sample(p).(VarList(q)) = reshape(sample(p).(VarList(q)),numPtsU,numPtsV);
    end
end

aspect = [10 1 1];
FontS = 10;
TextSize = 14;

s = 10; %PlotLength
for p=1:col
    for q=1:row
        subplot(row,col*s, [s*((q-1)*col+p-1)+1  s*((q-1)*col+p-1)+s])
        contourf(sample(p).X,sample(p).Y,sample(p).(VarList(q)),500,'edgecolor','none')
        if q ~= row
            set(gca,'Xticklabel',[])
        end
        if p ~= 1
            set(gca,'Yticklabel',[])
        end
        pbaspect(aspect)
        colormap jet
        %clim(limits(q,:));
        colorbar
        set(gca,'FontSize',FontS)
        title(titles(p,q),'Interpreter','latex','FontSize',TextSize)
    end
end

end

