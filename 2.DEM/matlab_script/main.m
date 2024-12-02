clc
clear
close all

%% Read Data
porosity_stateS = ["state1", "state2"];
BCstateS = ["C-C", "C-H", "H-H"];
porosityS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
beam_lengthS = [0.1, 0.2, 0.3 ,0.5, 1.0, 2.0, 3.0, 4.0, 5.0];

Datalist = ["porosity_state", "BCstate", "porosity", "beam_length", ...
            "xPhys", "yPhys", "u", "w", ...
            "eps_xx", "eps_yy", "eps_xy", ...
            "stress_xx", "stress_yy", "stress_xy", "loss"];

for Name = Datalist
    template.(Name) = [];
end


Data = repmat(template,length(porosity_stateS),...
                       length(BCstateS),...
                       length(porosityS),...
                       length(beam_lengthS) );

list = ["i","j","k","l"];

for i = 1:length(porosity_stateS)
    for j = 1:length(BCstateS)
        for k = 1:length(porosityS)
            for l = 1:length(beam_lengthS)

                file_path = "Data/" + BCstateS(j) + "_" + porosity_stateS(i) + "_" ...
                    + num2str(porosityS(k)) + "_" + num2str(beam_lengthS(l),'%.1f');


                for m = 1:4
                    Data(i,j,k,l).(Datalist(m)) = ...
                        evalin("caller", Datalist(m) + "S(" + list(m) + ")");
                end
                
                for DataName = Datalist(:,5:end)
                    fileID = fopen(file_path + DataName + ".txt" ,'r');
                    disp(file_path + DataName)
                    Data(i,j,k,l).(DataName) = fscanf(fileID,'%f');
                    fclose(fileID);
                end

            end
        end
    end
end

porosity_stateS = ["Symmetric", "Asymmetric"];

%% Type 1 - Figure 1

state = "Asymmetric";
BC = "C-H";
e0 = [0, 0.6];
L = 2.0;

H = 0.1;

i = find(ismember(porosity_stateS, state));
j = find(ismember(BCstateS, BC));
k = find(ismember(porosityS, e0));
l = find(ismember(beam_lengthS, L));

VarList = ["stress_xx", "stress_xy", "stress_yy"];
Vartitle =["sigma_{xx}", "sigma_{xz}", "sigma_{zz}"];

titles(length(e0),length(VarList)) = "";
for p = 1:length(e0)
    for q = 1:length(VarList)
        titles(p,q) = "$$\" + Vartitle(q) + " \;\;\; (e_0 = " + num2str(e0(p),'%.1f') +...
                      ", L/h = " + num2str(L/H) + ")$$, " + reverse(BC) + ", " + state; 
    end
end

clear limits
limits(1,:) = [-20,+20];
limits(2,:) = [-2,+2];
limits(3,:) = [-2,+2];

figure('Units','normalized','OuterPosition',[0 0 1 1])
make_fig1(Data, i,j,k,l, VarList, titles, limits)

%% Type 1 - Figure 2

state = "Symmetric";
BC = "C-C";
e0 = [0, 0.6];
L = 2.0;

H = 0.1;

i = find(ismember(porosity_stateS, state));
j = find(ismember(BCstateS, BC));
k = find(ismember(porosityS, e0));
l = find(ismember(beam_lengthS, L));

VarList = ["eps_xx", "eps_xy", "eps_yy"];
Vartitle =["varepsilon_{xx}", "varepsilon_{xz}", "varepsilon_{zz}"];

clear titles
titles(length(e0),length(VarList)) = "";
for p = 1:length(e0)
    for q = 1:length(VarList)
        titles(p,q) = "$$\" + Vartitle(q) + " \;\;\; (e_0 = " + num2str(e0(p),'%.1f') +...
                      ", L/h = " + num2str(L/H) + ")$$, " + reverse(BC) + ", " + state; 
    end
end

clear limits
limits(1,:) = [-20,+20];
limits(2,:) = [-2,+2];
limits(3,:) = [-2,+2];

figure('Units','normalized','OuterPosition',[0 0 1 1])
make_fig1(Data, i,j,k,l, VarList, titles, limits)

%% Type 1 - Figure 3

state = "Asymmetric";
BC = "H-H";
e0 = [0, 0.6];
L = 2.0;

H = 0.1;

i = find(ismember(porosity_stateS, state));
j = find(ismember(BCstateS, BC));
k = find(ismember(porosityS, e0));
l = find(ismember(beam_lengthS, L));

VarList = ["u", "w"];
Vartitle =["mathbf(u_x)", "mathbf(u_z)"];

clear titles
titles(length(e0),length(VarList)) = "";
for p = 1:length(e0)
    for q = 1:length(VarList)
        titles(p,q) = "$$\" + Vartitle(q) + " \;\;\; (e_0 = " + num2str(e0(p),'%.1f') +...
                      ", L/h = " + num2str(L/H) + ")$$, " + reverse(BC) + ", " + state; 
    end
end

clear limits
limits(1,:) = [-5e-5,+5e-5];
limits(2,:) = [-7e-4,0];

figure('Units','normalized','OuterPosition',[0 0 1 1])
make_fig1(Data, i,j,k,l, VarList, titles, limits)

%% Type 2 - Figure 4

state = ["Symmetric", "Asymmetric"];
BC = ["C-C", "C-H", "H-H"];
e0 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
L = [0.1, 0.2, 0.3 ,0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
%L = [0.1, 0.2, 0.3 ,0.5];
H = 0.1;

i = find(ismember(porosity_stateS, state));
j = find(ismember(BCstateS, BC));
k = find(ismember(porosityS, e0));
l = find(ismember(beam_lengthS, L));

VarList = "w";
Vartitle = "mathbf(w)";

clear titles
titles(length(state),length(BC)) = "";
    
for p = 1:length(state)
    for q = 1:length(BC)
        titles(p,q) = reverse(BC(q)) + ", " + state(p); 
    end
end

clear limits
limits{1,1} = [-0.03, 0];
limits{1,2} = [-0.05, 0];
limits{1,3} = [-0.07, 0];
limits{2,1} = [-0.04, 0];
limits{2,2} = [-0.06, 0];
limits{2,3} = [-0.1, 0];

lineStyles = {'b-', 'b--', 'b:', 'b-.','k-', 'k--', 'k:', 'k-.'};
Legends = {'$e_{0} = 0.1$','$e_{0} = 0.2$','$e_{0} = 0.3$','$e_{0} = 0.4$',...
           '$e_{0} = 0.5$','$e_{0} = 0.6$','$e_{0} = 0.7$','$e_{0} = 0.8$'};

figure('Units','normalized','OuterPosition',[0 0 1 1])
make_fig2(Data,L , i,j,k,l, VarList, titles, limits,lineStyles,Legends)


%% Type 3 - Figure 5


state = ["Symmetric", "Asymmetric"];
BC = ["C-H"];
e0 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
L = [0.1, 0.2, 0.3 ,0.5, 1.0, 2.0, 3.0, 4.0, 5.0];

H = 0.1;

i = find(ismember(porosity_stateS, state)); % p3
j = find(ismember(BCstateS, BC));           % q
k = find(ismember(porosityS, e0));          % r
l = find(ismember(beam_lengthS, L));        % s

VarList = "stress_xx";
Vartitle = "sigma_{xx}";

lineStyles = {'k-', 'b-', 'b--', 'b:', 'b-.','r-', 'r--', 'r:', 'r-.'};

% %%% should change
% clear titles
% titles(length(state),length(BC)) = "";
%     
% for p = 1:length(state)
%     for q = 1:length(BC)
%         titles(p,q) = reverse(BC(q)) + ", " + state(p); 
%     end
% end
% 
% clear limits
% limits{1,1} = [-0.03, 0];
% limits{1,2} = [-0.05, 0];
% limits{1,3} = [-0.07, 0];
% limits{2,1} = [-0.04, 0];
% limits{2,2} = [-0.06, 0];
% limits{2,3} = [-0.1, 0];
% %%% should change

figure('Units','normalized','OuterPosition',[0 0 1 1])
make_fig3(Data,L , i,j,k,l, VarList, titles, limits, lineStyles)








