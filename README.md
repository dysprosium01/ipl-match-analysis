**IPL Phase-Wise Tactical Match Analysis** 

A Data Analysis project that explores how matches are won and lost across different innings phases;
- PowerPlay overs (1-6)
- Middle Overs (7-15)
- Death Overs (16-20)
Instead of looking only at final scores and toss results, this project analyzes ball-by-ball data to uncover patterns in team performance, venue behavior, and player impact.

**Project Goal:**

Most IPL analysis focuses on:
- Average scores
- Toss outcomes
- Overall team statistics

This project tries to answer deeper questions:
- Does winning the toss really matter?
- Which phase contributes the most to winning?
- Which venues favor batters or bowlers?
- Who performs best under pressure?
- How much does a wicket affect scoring momentum?

**Tech Stack:**

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Streamlit

**Key Analytical Features:**

- Toss Impact Analysis:
A common belief in cricket is that winning the toss provides a significant advantage.
This part compares match outcomes across IPL seasons and shows that toss alone have a limited impact on winning. Team performance during different phases of the match plays a much larger role.

- Phase Impact on the match:
Each inning was divided into 3 different phases (Powerplay, Middle Overs, Death Overs). The contribution of each of these phase to the final score was calculated and compared between the winners and the losers of the matches.
It was found that teams that perform in the Middle Overs tend to win more matches.

- Top batters and Bowlers:
Analyzed batters in the Death Overs using metrics such as (Runs Scored, Balls faced, Strike Rate).  
Analyzed bowlers in the Powerplay phase using metrics such as (Wickets taken, Balls Bowled, Bowling Strike Rate).

- Venue Specific Analysis:
Found out stadiums favoring batters based on average run rate across different inning phases.
Found out stadiums favoring bowlers based on average bowling strike rates across different inning phases.

- Wicket Cascade Impact:
Measured the momentum shift in the run rate immediately after a wicket drops. After the wicket dropped the next 12 deliveries were analyzed to see how scoring changed. Most Teams often experienced a drop in their scoring rates after losing a wicket.

- Choke Hold Metric: 
Analyzed dot ball percentages across different venues across different phase innings to see where bowlers had complete control over batters. Plotted a heatmap for a visual comparison. 

**Conclusion:**
This project shows that IPL matches are influenced by much more than toss results and final scores. By analyzing the game phase-by-phase, we can better understand the factors that drive success in T20 cricket.
