function selectedIndividualIndex = TournamentSelect(fitnessList, tournamentProbability, tournamentSize)
    
    tournamentIndividuals = fitnessList(randperm(length(fitnessList), tournamentSize)); % Pick trournamentSize many individuals randomly
    sortedFitness = sort(tournamentIndividuals, 'descend');
    selectedIndividualIndex = find(fitnessList==sortedFitness(end), 1); % Assign index of last individual in sortedFitness if loop below does not "find" any better

    for j = 1:length(sortedFitness)
        r = rand;
        if r < tournamentProbability
            selectedIndividualIndex = find(fitnessList==sortedFitness(j), 1);
            break
        else
            continue
        end
    end
end