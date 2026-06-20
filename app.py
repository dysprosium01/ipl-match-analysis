import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="IPL Phase-Wise Tactical Analysis", layout="wide")
sns.set_theme(style="ticks", context="talk")
plt.style.use("dark_background")

# DATA LOADING
@st.cache_data
def load_data():
    df = pd.read_csv("ipl_dataset.csv", low_memory=False)
    def get_phase(over):
        if over < 6:
            return "Powerplay"
        elif over < 15:
            return "Middle Overs"
        else:
            return "Death Over"
    df["match_phase"] = df["over"].apply(get_phase)
    df["is_wicket"] = df["wicket_kind"].notna().astype(int)
    df["wicket_taken"] = df["wicket_player_out"].notna().astype(int)
    return df

# RUNNING ALL THE ANALYSIS

# TOSS ANALYSIS
@st.cache_data
def compute_toss_stats(df):
    total_bats = df[df['toss_decision'] == 'bat'].shape[0]
    total_bowls = df[df['toss_decision'] == 'field'].shape[0]

    total_bats_winners  = df[(df['toss_decision'] == 'bat')   & (df['toss_winner'] == df['winner'])].shape[0]
    total_bowl_winners  = df[(df['toss_decision'] == 'field') & (df['toss_winner'] == df['winner'])].shape[0]
    total_bats_losers   = df[(df['toss_decision'] == 'bat')   & (df['toss_winner'] != df['winner'])].shape[0]
    total_bowl_losers   = df[(df['toss_decision'] == 'field') & (df['toss_winner'] != df['winner'])].shape[0]

    toss_match_winners = df[df['toss_winner'] == df['winner']].shape[0]
    total_matches = df['winner'].count()

    toss_winner_win_rate = toss_match_winners * 100 / total_matches
    toss_loser_lose_rate = 100 - toss_winner_win_rate

    return (
        total_bats, total_bowls,
        total_bats_winners, total_bowl_winners,
        total_bats_losers, total_bowl_losers,
        toss_winner_win_rate, toss_loser_lose_rate
    )


# PHASE IMPACT ANALYSIS 
@st.cache_data
def compute_phase_stats(df):
    total_scores_runs = df.groupby(["match_id", "innings", "venue"])["runs_total"].sum().reset_index()
    phase_scores_runs = df.groupby(['match_id', 'innings', 'match_phase'])['runs_total'].sum().unstack().reset_index()

    merged_stats = pd.merge(phase_scores_runs, total_scores_runs, on=["match_id", "innings"])

    phase_cols = ['Powerplay', 'Middle Overs', 'Death Over']
    for col in phase_cols:
        if col not in merged_stats.columns:
            merged_stats[col] = 0
        else:
            merged_stats[col] = merged_stats[col].fillna(0)

    merged_stats['powerplay_pct']    = ((merged_stats['Powerplay']     / merged_stats['runs_total']) * 100).round(2)
    merged_stats['middle_overs_pct'] = ((merged_stats['Middle Overs']  / merged_stats['runs_total']) * 100).round(2)
    merged_stats['death_overs_pct']  = ((merged_stats['Death Over']    / merged_stats['runs_total']) * 100).round(2)

    inn1_scores = merged_stats[merged_stats['innings'] == 1][['match_id', 'runs_total']].rename(columns={'runs_total': 'inn1_total'})
    inn2_scores = merged_stats[merged_stats['innings'] == 2][['match_id', 'runs_total']].rename(columns={'runs_total': 'inn2_total'})

    merged_stats = pd.merge(merged_stats, inn1_scores, on='match_id', how='left')
    merged_stats = pd.merge(merged_stats, inn2_scores, on='match_id', how='left')

    def decision(row):
        if pd.isna(row['inn1_total']) or pd.isna(row['inn2_total']):
            return 'Tie'
        if row['innings'] == 1:
            return 'Won' if row['inn1_total'] > row['inn2_total'] else 'Lost'
        else:
            return 'Won' if row['inn2_total'] > row['inn1_total'] else 'Lost'

    merged_stats['outcome'] = merged_stats.apply(decision, axis=1)
    clean_df = merged_stats[merged_stats['outcome'].isin(['Won', 'Lost'])]

    final_summary = clean_df.groupby('outcome')[['powerplay_pct', 'middle_overs_pct', 'death_overs_pct']].mean()
    return final_summary


# BATTER & BOWLER ANALYSIS
@st.cache_data
def compute_batter_bowler_stats(df):
    venue = df.groupby(['match_phase', 'venue']).agg(phase_runs=('runs_total', 'sum'),phase_balls=('ball', 'count'),phase_wickets=('is_wicket', 'sum')).reset_index()

    venue['avg_run_rate']  = (venue['phase_runs'] * 6 / venue['phase_balls']).round(2)
    venue['avg_bowl_rate'] = (venue['phase_runs'] / venue['phase_wickets']).round(2)

    venue_batters = venue[['match_phase', 'venue', 'phase_runs', 'phase_balls', 'avg_run_rate']]
    venue_bowlers = venue[['match_phase', 'venue', 'phase_runs', 'phase_wickets', 'phase_balls', 'avg_bowl_rate']]
    venue_bowlers = venue_bowlers[(venue_bowlers['phase_balls'] >= 300) & (venue_bowlers['match_phase'] == "Powerplay")]

    # Batter stats
    df_batters = df.groupby(['season', 'match_phase', 'batter']).agg(total_runs=('runs_batter', 'sum'),total_balls=('ball', 'count')).reset_index()
    df_batters['batter_strike_rate'] = ((df_batters['total_runs'] / df_batters['total_balls']) * 100).round(2)
    df_batters_filtered = df_batters[df_batters['total_balls'] >= 30]

    # Bowler stats
    df_bowlers = df.groupby(['season', 'match_phase', 'bowler']).agg(total_wickets=('is_wicket', 'sum'),total_balls=('ball', 'count')).reset_index()
    df_bowlers['bowler_strike_rate'] = (df_bowlers['total_balls'] / df_bowlers['total_wickets']).round(2)
    df_bowlers_filtered = df_bowlers[(df_bowlers['total_balls'] >= 30) & (df_bowlers['total_wickets'] >= 3)]

    death_batters     = df_batters_filtered[df_batters_filtered['match_phase'] == "Death Over"].sort_values(by='batter_strike_rate', ascending=False).head(5)
    powerplay_bowlers = df_bowlers_filtered[df_bowlers_filtered['match_phase'] == "Powerplay"].sort_values(by='bowler_strike_rate', ascending=True).head(5)

    return venue, venue_batters, venue_bowlers, death_batters, powerplay_bowlers


# VENUE ANALYSIS
@st.cache_data
def compute_venue_stats(venue, venue_batters, venue_bowlers):
    phase_order = ['Powerplay', 'Middle Overs', 'Death Over']

    venue_volumes = venue_batters.groupby('venue')['phase_balls'].sum().reset_index()
    top_5_venues  = venue_volumes.sort_values(by='phase_balls', ascending=False).head(5)['venue'].tolist()

    df_top_5 = venue_batters[venue_batters['venue'].isin(top_5_venues)].copy()
    df_top_5['Stadium_name'] = df_top_5['venue'].apply(lambda x: x.split(',')[0])
    df_top_5 = df_top_5.drop_duplicates(subset=['Stadium_name', 'match_phase'])

    pivot_df = df_top_5.pivot(index='Stadium_name', columns='match_phase', values='avg_run_rate')
    pivot_df = pivot_df[phase_order]

    top_bowling_venues = venue_bowlers.sort_values(by='avg_bowl_rate', ascending=True).head(5)['venue'].tolist()
    df_top_5_bowlers2 = venue[venue['venue'].isin(top_bowling_venues)].copy()
    df_top_5_bowlers2['stadium_name'] = df_top_5_bowlers2['venue'].apply(lambda x: x.split(',')[0])
    df_top_5_bowlers2.drop_duplicates(subset=['stadium_name', 'match_phase'], inplace=True)

    pivot_df3 = df_top_5_bowlers2.pivot(index='stadium_name', columns='match_phase', values='avg_bowl_rate')
    pivot_df3 = pivot_df3[phase_order]

    return pivot_df, pivot_df3

# WICKET IMPACT ANALYSIS
@st.cache_data
def compute_wicket_impact(df):
    runs_after_wickets = []
    window_size = 12

    for i in range(len(df) - window_size):
        if df.iloc[i]['wicket_taken'] == 1:
            current_match = df.iloc[i]['match_id']
            next_balls = df.iloc[i+1: i+1+window_size]
            same_innings_balls = next_balls[next_balls['match_id'] == current_match]
            for run in same_innings_balls['runs_total']:
                runs_after_wickets.append(run)

    normal_pace = df['runs_total'].mean() * 6
    panic_pace  = np.mean(runs_after_wickets) * 6

    return runs_after_wickets, normal_pace, panic_pace

# DOT BALL ANALYSIS
@st.cache_data
def compute_dot_ball_stats(df):
    venue_phase_totals = df.groupby(['venue', 'match_phase']).size()
    venue_phase_dots = df[df['runs_total'] == 0].groupby(['venue', 'match_phase']).size()
    venue_dot_percentages = (venue_phase_dots / venue_phase_totals * 100).fillna(0)

    venue_analysis_df = venue_dot_percentages.unstack().fillna(0).reset_index()
    venue_analysis_df['Stadium Name'] = venue_analysis_df['venue'].apply(lambda x: x.split(',')[0])
    heatmap_data = venue_analysis_df.set_index('Stadium Name').drop(columns=['venue'])

    return heatmap_data


# RENDER UDFs FOR ALL THE VISUALIZATION CHARTS
def render_toss_analysis(toss_stats):
    (
        total_bats, total_bowls,
        total_bats_winners, total_bowl_winners,
        total_bats_losers, total_bowl_losers,
        toss_winner_win_rate, toss_loser_lose_rate
    ) = toss_stats

    st.header("Do teams that win the toss actually win more matches?")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Summary Statistics**")
        st.write(f"- Teams **winning** after choosing to **bat**: `{(total_bats_winners*100/total_bats):.2f}%`")
        st.write(f"- Teams **winning** after choosing to **bowl**: `{(total_bowl_winners*100/total_bowls):.2f}%`")
        st.write(f"- Teams **losing** after choosing to **bat**: `{(total_bats_losers*100/total_bats):.2f}%`")
        st.write(f"- Teams **losing** after choosing to **bowl**: `{(total_bowl_losers*100/total_bowls):.2f}%`")
        st.write(f"- Toss winners winning the match: `{toss_winner_win_rate:.2f}%`")
        st.write(f"- Toss losers winning the match: `{toss_loser_lose_rate:.2f}%`")
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        categories = ['Toss Winners', 'Toss Losers']
        vals = [toss_winner_win_rate, toss_loser_lose_rate]
        colors = ['#ff4d4d', '#2ecc71']
        bars = ax.bar(categories, vals, color=colors, width=0.2)
        ax.bar_label(bars, fmt='%.2f%%', fontweight='bold', fontsize=12, padding=5)
        ax.set_title('Win Rate : Toss Winners vs Toss Losers', fontsize=16, fontweight='bold')
        ax.set_xlabel('Toss Decision', fontsize=12, fontweight='bold')
        ax.set_ylabel('% Win Rate', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 80)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def render_phase_impact(final_summary):
    st.header("Which phase impacts victory the most — Powerplay, Middle Overs, or Death Overs?")

    plot_data = final_summary.reset_index()
    melted_data = pd.melt(plot_data, id_vars=['outcome'],
                          value_vars=['powerplay_pct', 'middle_overs_pct', 'death_overs_pct'],
                          var_name='Match Phase', value_name='Average Percentage of Total Score')
    melted_data['Match Phase'] = melted_data['Match Phase'].replace({
        'powerplay_pct':    'Powerplay (Overs 1-6)',
        'middle_overs_pct': 'Middle Overs (Overs 7-15)',
        'death_overs_pct':  'Death Overs (Overs 16-20)'
    })

    fig, ax = plt.subplots(figsize=(15, 6))
    sns.barplot(data=melted_data, x='Match Phase', y='Average Percentage of Total Score',
                hue='outcome', palette=['#ff4d4d', '#2ecc71'], ax=ax, gap=1.5)
    ax.set_title('Which Phase Impacts the Match Outcome the Most?', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Tournament Match Phases', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg % Contribution to Team Score', fontsize=12, fontweight='bold')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10, fontweight='bold')
    ax.legend(title='Match Outcome', loc='upper right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("**Key Insight:** Fans focus on the Powerplay for boundaries and the Death Overs for finishes, but the Middle Overs (Overs 6–14) win T20 matches. The start and end phases are highly standardized and often cancel each other out. The middle overs are where the game slows down, dot-ball pressure builds, and tactical errors occur. Teams that control the middle overs win over 44% of games.")


def render_batter_bowler_charts(death_batters, powerplay_bowlers):
    st.header("Who are the top batters and bowlers across seasons?")

    fig, ax = plt.subplots(1, 2, figsize=(20, 6))

    bars_batters = ax[0].barh(death_batters['batter'], death_batters['batter_strike_rate'], color="#b67424")
    ax[0].set_title("Top Death Phase Batters", fontsize=15, fontweight='bold')
    ax[0].set_xlabel("Batter's Strike Rate", fontsize=15, fontweight="bold")
    ax[0].set_ylabel("Batters", fontsize=15, fontweight="bold")
    ax[0].bar_label(bars_batters, fmt='%.2f', padding=5, fontsize=12, fontweight='bold', color="#dde5ed")
    ax[0].set_xlim(0, max(death_batters['batter_strike_rate']) * 1.10)

    bars_bowlers = ax[1].barh(powerplay_bowlers['bowler'], powerplay_bowlers['bowler_strike_rate'], color="#2980b9")
    ax[1].set_title("Top Powerplay Bowlers", fontsize=15, fontweight="bold")
    ax[1].set_xlabel("Bowler's Strike Rate", fontsize=15, fontweight="bold")
    ax[1].set_ylabel("Bowlers", fontsize=15, fontweight="bold")
    ax[1].bar_label(bars_bowlers, fmt='%.2f', padding=5, fontsize=12, fontweight='bold', color='#2c3e50')
    ax[1].set_xlim(0, max(powerplay_bowlers['bowler_strike_rate']) * 1.10)

    plt.suptitle("Top 5 Batters and Bowlers Across the Season", fontsize=20, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_venue_analysis(pivot_df, pivot_df3):
    st.header("Venue Specific Phase Analysis")

    st.subheader("Pitch Analysis: Profitable Stadiums for Batters")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pivot_df.T, marker='o', linewidth=2.5)
    ax.set_title('Pitch Analysis: Profitable Stadiums for Batters', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Match Phases', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Run Rate', fontsize=11, fontweight='bold')
    ax.legend(pivot_df.index, title='High-Volume Stadiums', loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.info("**Key Insight:** Some stadiums naturally make batting easier and lead to higher scoring rates. By identifying these batter-friendly venues, teams can understand where aggressive batting is more rewarding and where playing too cautiously may waste an advantage. \nInstead of relying on slow starts or conservative anchor roles, teams can adapt their strategy to attack earlier, take more scoring risks, and fully use the conditions in their favor. This helps captains and coaches make smarter decisions about batting order, intent, and overall match tempo based on the venue itself.")

    st.subheader("Pitch Analysis: Profitable Stadiums for Bowlers")
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(pivot_df3.T, marker='o', linewidth=2.5)
    ax.set_title('Pitch Analysis: Profitable Stadiums for Bowlers', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Match Phases', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Bowling Rate', fontsize=11, fontweight='bold')
    ax.legend(pivot_df3.index, title='High-Volume Stadiums', loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.info("**A Special Case: The Nehru Stadium** — during the middle 8-9 overs, a massive spike to ~37.5 suggests that batters are stuck rotating singles and doubles, forcing desperation in the death overs.\n\n**DY Patil & Newlands** suggest batters can score in the powerplay (larger fields), but the massive spike down in bowling rate during death overs becomes a graveyard for batters forced to hit longer shots.")

def render_wicket_impact(normal_pace, panic_pace, window_size=12):
    st.header("Wicket Impact On Run Rate")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Normal Match Pace (RPO)", f"{normal_pace:.2f}")
    with col2:
        st.metric(f"Post-Wicket Pace — Next {window_size} Balls (RPO)", f"{panic_pace:.2f}")

    categories = ['Normal Match Pace', f'Post-Wicket Pace\n(Next {window_size} Balls)']
    paces  = [normal_pace, panic_pace]
    colors = ['#1f77b4', '#d62728']

    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(categories, paces, color=colors, width=0.4)
    ax.bar_label(bars, fmt='%.2f RPO', padding=3, weight='bold', fontsize=11)
    ax.set_ylabel("Runs Per Over (RPO)", fontsize=12)
    ax.set_title(f"Innings Momentum Shock Window (Window: {window_size} Balls)", fontsize=13, pad=15, weight='bold')
    ax.set_ylim(0, max(paces) + 2)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def render_dot_ball_heatmap(heatmap_data):
    st.header("Choke Hold Metric — Where Do Bowlers Bowl the Most Dot Balls?")

    fig, ax = plt.subplots(figsize=(20, 8))
    sns.heatmap(heatmap_data, ax=ax, cmap='viridis')
    ax.set_title("Where Do Bowlers Suffocate Teams the Most?", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Match Phase", fontweight='bold', fontsize=16)
    ax.set_ylabel("Stadium Venue", fontweight='bold', fontsize=16)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.info("**Key Insight:** Overall stadium averages don't tell the full story of how a match unfolds. A pitch can behave very differently across the Powerplay, Middle Overs, and Death Overs. Our Choke-Hold Index looks at dot-ball percentages in each phase to identify when bowling teams are most effective at slowing down the scoring rate. This gives teams a clearer, phase-by-phase understanding of where pressure builds and how bowling units can control the momentum of an innings.")


# MAIN APP
def main():
    st.title("Phase-Wise Tactical IPL Match Analysis")
    st.markdown(
        "A data analytics project analyzing IPL T20 matches using ball-by-ball data to uncover tactical insights beyond traditional stadium averages."
    )

    # Load data once
    df = load_data()

    # Run all analysis (cached)
    toss_stats = compute_toss_stats(df)
    final_summary = compute_phase_stats(df)

    venue, venue_batters, venue_bowlers, \
        death_batters, powerplay_bowlers = compute_batter_bowler_stats(df)

    pivot_df, pivot_df3 = compute_venue_stats(
        venue,
        venue_batters,
        venue_bowlers
    )

    _, normal_pace, panic_pace = compute_wicket_impact(df)
    heatmap_data = compute_dot_ball_stats(df)

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Toss Analysis",
        "Phase Impact",
        "Players",
        "Venue Analysis",
        "Wicket Impact",
        "Choke Hold"
    ])

    with tab1:
        render_toss_analysis(toss_stats)

    with tab2:
        render_phase_impact(final_summary)

    with tab3:
        render_batter_bowler_charts(death_batters,powerplay_bowlers)

    with tab4:
        render_venue_analysis(pivot_df,pivot_df3)

    with tab5:
        render_wicket_impact(normal_pace,panic_pace)

    with tab6:
        render_dot_ball_heatmap(heatmap_data)

if __name__ == "__main__":
    main()