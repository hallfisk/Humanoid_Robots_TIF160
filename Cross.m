function [newIndividual1, newIndividual2] = Cross(individual1, individual2, maxChromosomeLength)
    
    nrOfGenes1 = length(individual1);
    nrOfGenes2 = length(individual2);

    crossOverPointsIndividual1 = sort(4*randi([1, fix(nrOfGenes1/4)], 1, 2));
    crossOverPointsIndividual2 = sort(4*randi([1, fix(nrOfGenes2/4)], 1, 2));
    
    % Create new individuals based on figure 3.21 in course book (pp.77)
    newIndividual1 = [individual1(1:crossOverPointsIndividual1(1)) individual2(crossOverPointsIndividual2(1):crossOverPointsIndividual2(2)) individual1(crossOverPointsIndividual1(2):end)];
    newIndividual2 = [individual2(1:crossOverPointsIndividual2(1)) individual1(crossOverPointsIndividual1(1):crossOverPointsIndividual1(2)) individual2(crossOverPointsIndividual2(2):end)];
    
    % Trim or truncate new individuals if they exceed MaxChromosomeLength
    newIndividual1 = TrimChromosome(newIndividual1, maxChromosomeLength);
    newIndividual2 = TrimChromosome(newIndividual2, maxChromosomeLength);
end

function trimmedChromosome = TrimChromosome(chromosome, maxChromosomeLength)
    if length(chromosome) > maxChromosomeLength
        trimmedChromosome = chromosome(1:maxChromosomeLength);
    else
        trimmedChromosome = chromosome;
    end
end