# onion-universe

This is my code repository for anything related to the finite coasting cosmology, also known (by me ;^D) as "the onion universe". Strictly speaking, such alias is most appropriate for the $R(t) = ct$ case ($\Omega_k = -1$), but this is (once again, in my opinion) the only model that makes sense close to the Big Bang, and I hope I can eventaually recover it as $t \to 0$. In the meantime, I guess I can live with any other value of the curvature density, as long as it is negative, and there is no repulsive cosmological constant ;^D

Have fun,

Yago

                                                           ...Paranoy@ Rulz! ;^D


PD: The files in this repository are organised as follows:
    1.spacetime.ipnyb -> Visual representations of Minkowski diagrams for the onion model against LCDM. Only for visualization purpouses, no real analysis.
    2.evolution.ipnyb -> Comparison of theoretical measures for the cosmological distances of onion model against LCDM (first part correspond to old versionns of the model, skip to second half).
    3.fitting.ipnyb   -> Main file for the analysis. First part loads the data from the collaborations used in the posterior analysis, then we estimate the parameters in our onion and LCDM models using a grid and chi2 fits of the mdoels and data, and finally we calculate the goodness of fit for each model.

    models_geo.py     -> Definition of the class and functions for the computation of distances and chi2 fits used in the rest of the code
    models_rd.py      -> Corresponds to an old version of the model IGNORE!! 
    models.py         -> Corresponds to an old version of the model IGNORE!! 
