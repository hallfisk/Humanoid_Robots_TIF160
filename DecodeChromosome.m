function x = DecodeChromosome(chromosome,numberOfVariables,maximumVariableValue)
    
    x = zeros(1, numberOfVariables);
    lenghtOfChromosome = length(chromosome);
    lenghtPerVariable = lenghtOfChromosome/numberOfVariables;
    cutChromosomes = reshape(chromosome,lenghtPerVariable,[]); %cutChromosomes(:,1) is the first cut chromosome (g_1 ... g_k)

    for i = 1:numberOfVariables
        factorTwoTermTwo = 0;
        for k = 1:lenghtPerVariable
            factorTwoTermTwo = factorTwoTermTwo + 2^(-k)*cutChromosomes(k,i);
        end

        factorOneTermTwo = (2*maximumVariableValue / (1-2^(-lenghtPerVariable)) );
        x(i) = -maximumVariableValue + factorOneTermTwo*factorTwoTermTwo;
    end

end