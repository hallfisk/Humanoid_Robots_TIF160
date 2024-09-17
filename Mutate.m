function mutatedIndividual = Mutate(individual, mutationProbability, nrOfOperators, nrOfVariableRegisters, nrOfConstantRegisters);
    
    nGenes = size(individual,2);
    mutatedIndividual = individual;

    nrOfOperands = nrOfVariableRegisters + nrOfConstantRegisters;
    
    for j = 1:nGenes
        r = rand;
        geneClass = mod(j,4);
        if r < mutationProbability
            if geneClass == 1
                mutatedIndividual(j) = randi(nrOfOperators);
            elseif geneClass == 2
                mutatedIndividual(j) = randi(nrOfVariableRegisters);
            else
                mutatedIndividual(j) = randi(nrOfOperands);
            end
        end
    end
end
