Used an ensemble of a few classification models to predict players chances of scoring above 5 points in the coming FPL GW. 
---
Limitations:
I can only find a complete dataset for this season, and as such the model is only trained on 20 gameweeks worth of data. \
The model can't predict individual player match ups.\
The dataset doesn't contain some data which could be important:\
Yellow cards, Red cards, Own goals, Is penatly taker, bonus points.

Improvements:
Integrate a data pipeline to get last gameweeks player data automatically.\
Adjust points to be scaled positionally, ie goals by defenders are weighted more highly.\
Integrate possession value models such as xT.\
Create a library for coming gameweeks to look further into the future.
