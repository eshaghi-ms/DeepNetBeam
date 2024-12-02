function Data = read_data(porosity_stateS, BCstateS, porosityS, beam_lengthS)

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


