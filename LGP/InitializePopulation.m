function population = InitializePopulation(populationSize, minChromosomeLength,maxChromosomeLength, nrOfVariableRegisters, nrOfConstantRegisters, nrOfOperators)

    population = [];
    nrOfOperands = nrOfVariableRegisters + nrOfConstantRegisters;
    
    for i = 1:populationSize
        chromosomeLength = minChromosomeLength + fix(rand*(maxChromosomeLength-minChromosomeLength+1));
        chromosome = zeros(1, chromosomeLength);
        for j = 1:4:chromosomeLength
            chromosome(j) = randi(nrOfOperators);
            chromosome(j+1) = randi(nrOfVariableRegisters);
            chromosome(j+2) = randi(nrOfOperands);
            chromosome(j+3) = randi(nrOfOperands);
        end
        
        individual = struct('Chromosome', chromosome);
        population = [population individual];
    end
          
end
