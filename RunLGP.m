clc;

% Define parameters and constants
populationSize = 100;
nrOfGenerations = 20000;
tournamentSize = 4;
tournamentProbability = 0.75;    
crossoverProbability = 0.7;

minMutationProbability = 0.02;
maxMutationProbability = 0.1;
alpha = 1.1;
minDiversity = 0.3;

minChromosomeLength = 20;
maxChromosomeLength = 100;

constantRegister = [1 -1 -3];
nrOfConstantRegisters = length(constantRegister);
nrOfVariableRegisters = 3;
nrOfOperators = 4; % (+, -, *, /)
nrOfOperands = nrOfVariableRegisters + nrOfConstantRegisters;

% Load data from LoadFunctionData.m
functionData = LoadFunctionData;

globalBestChromosome = [];
globalBestFitness = 0;
maxGenerationsWithoutImprovement = 5000;
nrOfRuns = 10000;

for run = 1:nrOfRuns
    
    % Initialize population with random chromosomes
    population = InitializePopulation(populationSize, minChromosomeLength, maxChromosomeLength, nrOfVariableRegisters, nrOfConstantRegisters, nrOfOperators);
    
    fitnessList = zeros(populationSize, 1);
    bestChromosome = [];
    bestFitness = 0;
    mutationProbability = minMutationProbability;
    maxGenerationsWithoutImprovementCounter = 0;

    for generation = 1:nrOfGenerations
        for i= 1:populationSize
            chromosome = population(i).Chromosome;
            fitnessList(i) = EvaluateIndividual(chromosome, functionData, constantRegister, nrOfVariableRegisters);
    
            if fitnessList(i)>bestFitness
                iBestIndividual = i;
                bestFitness = fitnessList(i);
                bestChromosome = chromosome;
            end
        end

    
        % Plot every 100 generations
        if mod(generation,100)==0
             nrOfDataPoints = length(functionData);
             bestEstimatedDataPoints = zeros(nrOfDataPoints, 1);
             globalBestEstimatedDataPoints = zeros(nrOfDataPoints, 1);

             for i = 1:nrOfDataPoints
                 x = functionData(i, 1);
                 bestEstimate = CalculateEstimation(bestChromosome, x, nrOfVariableRegisters, constantRegister);
                 globalBestEstimate = CalculateEstimation(globalBestChromosome, x, nrOfVariableRegisters, constantRegister);
                 bestEstimatedDataPoints(i) = bestEstimate;
                 globalBestEstimatedDataPoints(i) = globalBestEstimate;
             end
             plot(functionData(:, 1), bestEstimatedDataPoints, 'r'); % Current prediction
             hold on
             plot(functionData(:, 1), globalBestEstimatedDataPoints, 'g'); % Best prediction so far
             hold on
             scatter(functionData(:, 1), functionData(:, 2), 'b'); % Actual data
             hold off
             xlabel('x');
             ylabel('g(x)');
             legend('Current prediction', 'Best prediction so far', 'Function data', 'Location','southeast');
             title(['Run: ' num2str(run) ', Generation: ' num2str(generation)]);
             drawnow
        end

        if maxGenerationsWithoutImprovementCounter == maxGenerationsWithoutImprovement
            fprintf('No improvements for %d generations -> New run \n', maxGenerationsWithoutImprovement) 
            break;
        end
     
        temporaryPopulation = population;
        for i = 1:2:populationSize
            i1 = TournamentSelect(fitnessList,tournamentProbability,tournamentSize);
            i2 = TournamentSelect(fitnessList,tournamentProbability,tournamentSize);
            r = rand;
            if (r < crossoverProbability) 
                individual1 = population(i1).Chromosome;
                individual2 = population(i2).Chromosome;
                [newIndividual1, newIndividual2] = Cross(individual1, individual2, maxChromosomeLength);
                temporaryPopulation(i).Chromosome = newIndividual1;
                temporaryPopulation(i+1).Chromosome = newIndividual2;
            else
                temporaryPopulation(i).Chromosome = population(i1).Chromosome;
                temporaryPopulation(i+1).Chromosome = population(i2).Chromosome;     
            end
        end
       
        temporaryPopulation(1).Chromosome = population(iBestIndividual).Chromosome;
        for i = 2:populationSize
            individual = temporaryPopulation(i).Chromosome;
            tempIndividual = Mutate(individual, mutationProbability, nrOfOperators, nrOfVariableRegisters, nrOfConstantRegisters);
            temporaryPopulation(i).Chromosome = tempIndividual;
        end
    
        % Adjust mutation rate
        diversity = CalculateDiversity(population, populationSize, nrOfOperators, nrOfVariableRegisters, nrOfOperands);
        if diversity < minDiversity % minDiversity = maxDiversity is used here
            mutationProbability = mutationProbability*alpha;
        else
            mutationProbability = mutationProbability/alpha;
        end
        mutationProbability = max(mutationProbability, minMutationProbability);
        mutationProbability = min(mutationProbability, maxMutationProbability);
     
        population = temporaryPopulation;

        if bestFitness > globalBestFitness
            globalBestFitness = bestFitness;
            globalBestChromosome = bestChromosome;
            maxGenerationsWithoutImprovementCounter = 0;
        else
            maxGenerationsWithoutImprovementCounter = maxGenerationsWithoutImprovementCounter + 1;
        end
    
    end

end

% Store the best chromosome
save('BestChromosome.m', 'globalBestChromosome');
