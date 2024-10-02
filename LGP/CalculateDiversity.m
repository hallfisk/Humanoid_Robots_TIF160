function diversity = CalculateDiversity(population, populationSize, nrOfOperators, nrOfVariableRegisters, nbrOperands)
    totalDissimilarity = 0;

    % Use eq. (3.49) and (3.50) in course book
    for i = 1:populationSize-1
        chromosome1 = population(i).Chromosome;
        nrOfGenes1 = length(chromosome1);

        for j = i+1:populationSize
            chromosome2 = population(j).Chromosome;
            nrOfGenes2 = length(chromosome2);

            minNrOfGenes = min(nrOfGenes1, nrOfGenes2);

            dissimilarity = 0;
            for k = 1:minNrOfGenes
                diff = chromosome1(k) - chromosome2(k);
                diff = abs(diff);
                geneClass = mod(k,4);
                if geneClass == 1
                    Rk = nrOfOperators;
                elseif geneClass == 2
                    Rk = nrOfVariableRegisters;
                else
                    Rk = nbrOperands;
                end
                dissimilarity = dissimilarity + diff/Rk;
            end

            dissimilarity = dissimilarity / minNrOfGenes;

            % Accumulate dissimilarity for this pair of chromosomes
            totalDissimilarity = totalDissimilarity + dissimilarity;
        end
    end

    % Calculate diversity as the normalized average dissimilarity
    diversity = 2 * totalDissimilarity / (populationSize * (populationSize - 1));
end

