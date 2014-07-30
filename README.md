Businesses You May Know
===
Recommender engine for existing users of a social networking startup to invite other users to join the platform.


Townsquared is building private online communities for local businesses. The purpose of this project is to help increase the user base of TownSquared using a recommendation engine that prompts existing users to invite other businesses they might know to join the platform. 


Data sources used include information provided by TownSquared on facebook interactions of businesses and business information. 

    DonorsChoose Hacking Education
    National Center for Education Statistics
    2010 US Census school districtsl

Process

    Data pipeline
        get_donorschoose.py
        get_nces.py
        get_census.py
        get_latlon.py
    EDA on California schools alone
        california.py
    Train classifiers to predict DonorsChoose Activity
        feature_importance.py
    Explore feature-importances with the aggregated data
    Develop a fast cosine-similarity calculation using matrix multiplication
        similarity.py
    Recommend districts based on their cosine-similarity to active schools
        district.py
    Develop d3.js interactive visualization to explore result
        hosted on abshinn.github.io
        the code lives here

My first approach was to train classifiers to predict active DonorsChoose schools. Due to the complicated nature of the learning objective, it was difficult to predict activity with meaningful accuracy. As an alternative, I used the classifiers to whittle down the large feature set for a cosine-similarity calculation. The objective for the cosine-similarity calculation is to leverage the district-level aggregated data to find districts that are economically similar to the most active DonorsChoose schools.
End Product