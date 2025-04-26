import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
import sys

# load dataset
trades = pd.read_csv("trades.csv")

# refine data
trades.replace('STL', 'LAR', inplace=True)
trades.replace('LA', 'LAR', inplace=True)
trades.replace('OAK', 'LV', inplace=True)
trades.replace('SD', 'LAC', inplace=True)
trades = trades.dropna(subset=['pick_number'])
trades['pick_round'] = trades['pick_round'].astype('Int64')
trades['pick_number'] = trades['pick_number'].astype('Int64')

# for calculating value of a trade
BIN_SIZE  = 16
max_pick  = int(trades['pick_number'].max())
NUM_BINS  = math.ceil(max_pick / BIN_SIZE)

# mapping of every NFL team’s current abbrev. to its division, city and nickname
TEAM_INFO = {
    'ARI': {'division': 'NFC West',  'city': 'Arizona',      'name': 'Cardinals'},
    'ATL': {'division': 'NFC South', 'city': 'Atlanta',      'name': 'Falcons'},
    'BAL': {'division': 'AFC North', 'city': 'Baltimore',    'name': 'Ravens'},
    'BUF': {'division': 'AFC East',  'city': 'Buffalo',      'name': 'Bills'},
    'CAR': {'division': 'NFC South', 'city': 'Carolina',     'name': 'Panthers'},
    'CHI': {'division': 'NFC North', 'city': 'Chicago',      'name': 'Bears'},
    'CIN': {'division': 'AFC North', 'city': 'Cincinnati',   'name': 'Bengals'},
    'CLE': {'division': 'AFC North', 'city': 'Cleveland',    'name': 'Browns'},
    'DAL': {'division': 'NFC East',  'city': 'Dallas',       'name': 'Cowboys'},
    'DEN': {'division': 'AFC West',  'city': 'Denver',       'name': 'Broncos'},
    'DET': {'division': 'NFC North', 'city': 'Detroit',      'name': 'Lions'},
    'GB':  {'division': 'NFC North', 'city': 'Green Bay',    'name': 'Packers'},
    'HOU': {'division': 'AFC South', 'city': 'Houston',      'name': 'Texans'},
    'IND': {'division': 'AFC South', 'city': 'Indianapolis', 'name': 'Colts'},
    'JAX': {'division': 'AFC South', 'city': 'Jacksonville', 'name': 'Jaguars'},
    'KC':  {'division': 'AFC West',  'city': 'Kansas City',  'name': 'Chiefs'},
    'LAC': {'division': 'AFC West',  'city': 'Los Angeles',  'name': 'Chargers'},
    'LAR': {'division': 'NFC West',  'city': 'Los Angeles',  'name': 'Rams'},
    'LV':  {'division': 'AFC West',  'city': 'Las Vegas',    'name': 'Raiders'},
    'MIA': {'division': 'AFC East',  'city': 'Miami',        'name': 'Dolphins'},
    'MIN': {'division': 'NFC North', 'city': 'Minnesota',    'name': 'Vikings'},
    'NE':  {'division': 'AFC East',  'city': 'New England',  'name': 'Patriots'},
    'NO':  {'division': 'NFC South', 'city': 'New Orleans',  'name': 'Saints'},
    'NYG': {'division': 'NFC East',  'city': 'New York',     'name': 'Giants'},
    'NYJ': {'division': 'AFC East',  'city': 'New York',     'name': 'Jets'},
    'PHI': {'division': 'NFC East',  'city': 'Philadelphia', 'name': 'Eagles'},
    'PIT': {'division': 'AFC North', 'city': 'Pittsburgh',   'name': 'Steelers'},
    'SEA': {'division': 'NFC West',  'city': 'Seattle',      'name': 'Seahawks'},
    'SF':  {'division': 'NFC West',  'city': 'San Francisco','name': '49ers'},
    'TB':  {'division': 'NFC South', 'city': 'Tampa Bay',    'name': 'Buccaneers'},
    'TEN': {'division': 'AFC South', 'city': 'Tennessee',    'name': 'Titans'},
    'WAS': {'division': 'NFC East',  'city': 'Washington',   'name': 'Commanders'},
}

class Team:
    """
    Representation of an NFL team.

    Attributes:
        id (str): Current franchise abbreviation (e.g., 'ATL').
        division (str): Division name (e.g., 'NFC South').
        city (str): Team's city or region name (e.g., 'Atlanta').
        name (str): Team's nickname (e.g., 'Falcons').
    """

    def __init__(self, abbr):
        """
        Initialize a Team instance.

        Args:
            abbr (str): The current franchise abbreviation (e.g., 'ATL', 'LAR').

        Raises:
            ValueError: If the abbreviation is unrecognized.
        """
        info = TEAM_INFO.get(abbr)
        if not info:
            raise ValueError(f"Unknown team abbreviation: {abbr}")
        self.id = abbr
        self.division = info['division']
        self.city = info['city']
        self.name = info['name']

    def __eq__(self, other):
        """
        Compare equality with another Team instance.

        Two Team objects are equal if they have the same abbreviation.

        Args:
            other (Team): The object to compare against.

        Returns:
            bool: True if both are Team instances with identical abbreviations.
        """
        return isinstance(other, Team) and self.id == other.id

    def __hash__(self):
        """
        Return a hash based on the team's unique abbreviation.
        This makes Team instances usable as dict keys or graph nodes.
        """
        return hash(self.id)

    def __str__(self):
        """
        Return a human-readable string representation of the team.

        Returns:
            str: A string in the format 'City Name (Division)',
                 e.g., 'Atlanta Falcons (NFC South)'.
        """
        return f"{self.city} {self.name} ({self.division})"

class Trade:
    """
    Representation of a single NFL draft pick trade.

    Attributes:
        gave (Team): Team that gave the pick.
        received (Team): Team that received the pick.
        season (int): Season year when the trade occurred.
        pick_round (int): Round number of the pick.
        pick_number (int): Overall pick number.
        value (int): Binned and inverted value of pick quality (higher is better).
    """

    def __init__(self, gave, received, season, pick_round, pick_number):
        """
        Initialize a Trade instance.

        Args:
            gave (str): Abbreviation of team giving the pick.
            received (str): Abbreviation of team receiving the pick.
            season (int): Season year of the trade.
            pick_round (int): Round number of the pick.
            pick_number (int): Overall pick number.
        """
        self.gave = Team(gave)
        self.received = Team(received)
        self.season = season
        self.pick_round = pick_round
        self.pick_number = pick_number

        bin_index = (self.pick_number - 1) // BIN_SIZE
        self.value = NUM_BINS - bin_index

    def __str__(self):
        """
        Return a human-readable representation of the trade.

        Returns:
            str: A string describing the trade, including teams involved,
                 round, pick, and computed value.
        """
        return (
            f"{self.gave} → {self.received}, "
            f"round {self.pick_round} pick {self.pick_number} "
            f"→ value {self.value}"
        )

def build_graph(trades):
    """
    Construct a directed multigraph of NFL draft pick trades.

    Each node in the graph is a Team instance. An edge from node A to B
    represents a trade where team A gave a pick to team B. Multiple trades
    between the same teams are preserved as parallel edges.

    Args:
        trades (pd.DataFrame): DataFrame of trade records. Must include columns
            'gave', 'received', 'season', 'pick_round', 'pick_number', and 'trade_id'.

    Returns:
        nx.MultiDiGraph: A graph where:
            - Nodes are Team objects.
            - Each edge has attributes:
                * 'trade' (Trade): The Trade instance for that edge.
                * 'weight' (int): The pick value computed by Trade.value.
            - Edge keys correspond to the original trade_id, ensuring uniqueness.
    """
    G = nx.MultiDiGraph()
    for _, row in trades.iterrows():
        giver = Team(row['gave'])
        recv = Team(row['received'])

        t = Trade(
            row['gave'], row['received'],
            row['season'], row['pick_round'],
            row['pick_number']
        )
        G.add_edge(
            giver,
            recv,
            key=row['trade_id'],  # so each edge is unique
            trade=t,  # your Trade instance
            weight=t.value
        )

    return G

def display_graph(G):
    """
    Render and save a visualization of the trade network graph.

    Uses a spring layout to position nodes, draws each Team node and label,
    and renders directed edges as curved arrows with width proportional to weight.
    Saves the figure to 'trade_network.png' and displays it.

    Args:
        G (nx.MultiDiGraph): Graph built by build_graph, with Team nodes and
                             edges having 'weight' and 'trade' attributes.

    Returns:
        None
    """
    pos = nx.spring_layout(G, seed=42)

    nx.draw_networkx_nodes(G, pos, node_size=800, node_color='lightblue')
    labels = {team: team.name for team in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=9)

    for (u, v, key, data) in G.edges(keys=True, data=True):
        raw = (key + 1) * 0.1
        rad = max(min(raw, 1.0), -1.0)
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle='->',
            arrowsize=8,
            width=data['weight'] / 50  # or any scaling
        )

    plt.axis('off')
    plt.tight_layout()
    plt.savefig("nfl_trade_network.png", dpi=300, bbox_inches="tight")
    plt.show()

def most_related(team, G):
    """Find and print the team(s) with which a given team has the most trades."""
    try:
        team_obj = Team(team)
    except ValueError as e:
        print(f"↳ {e}")
        return

        # count trades between this team and every other
    counts = {}
    for u, v, _ in G.edges(keys=True):
        if u == team_obj:
            counts[v] = counts.get(v, 0) + 1
        elif v == team_obj:
            counts[u] = counts.get(u, 0) + 1

    if not counts:
        print(f"No trades found for {team_obj.id}.")
        return

    max_trades = max(counts.values())
    partners = [t for t, cnt in counts.items() if cnt == max_trades]

    print(f"Team(s) most related to {team_obj} ({max_trades} trades):")
    for t in partners:
        print(f"  • {t}")

def team_trades(team, season, G):
    """Count and print the number of trades a team participated in for a season."""
    try:
        team_obj = Team(team)
    except ValueError as e:
        print(f"↳ {e}")
        return

    if season < 2002 or season > 2023:
        print("Season must be in range 2002-2023.")
        return

    cnt = 0
    for _, _, _, data in G.edges(keys=True, data=True):
        tr = data['trade']
        if tr.season == season and (tr.gave == team_obj or tr.received == team_obj):
            cnt += 1

    print(f"{team_obj} participated in {cnt} trade(s) in the {season} season.")

def most_connected(G):
    """Identify and print the team with the most total trades"""
    degs = {team: G.degree(team) for team in G.nodes()}
    if not degs:
        print("Graph is empty.")
        return

    max_deg = max(degs.values())
    hubs = [t for t, d in degs.items() if d == max_deg]

    print(f"Most connected team ({max_deg} trades):")
    for t in hubs:
        print(f"  • {t}")

def highest_received(team, G):
    """For a specified team, print the trade(s) where it received the highest value."""
    try:
        team_obj = Team(team)
    except ValueError as e:
        print(f"↳ {e}")
        return

        # collect all trades where this team is the receiver
    incoming = [data['trade'] for _, v, _, data in G.edges(keys=True, data=True) if v == team_obj]

    if not incoming:
        print(f"{team_obj.id} never received a pick.")
        return

    max_val = max(tr.value for tr in incoming)
    top = [tr for tr in incoming if tr.value == max_val]

    print(f"Highest-value trade(s) received by {team_obj}:")
    for tr in top:
        print(f"  • {tr}")

def main():
    G = build_graph(trades)
    display_graph(G)

    menu = """
    Please choose an option (1–5):
     1) Find most closely related team to a given team
     2) Count how many trades a team had in a season
     3) Identify the most connected team
     4) For a team, show the trade(s) where they received the highest value
     5) Exit
    """

    while True:
        print(menu)
        choice = input("Enter choice: ").strip()

        if choice == '1':
            team = input("Enter team abbreviation (e.g. ATL): ").strip().upper()
            most_related(team, G)

        elif choice == '2':
            team = input("Enter team abbreviation (e.g. ATL): ").strip().upper()
            season = input("Enter season (2002-2023): ").strip()
            try:
                season = int(season)
            except ValueError:
                print("↳ Invalid season; must be an integer.\n")
                continue
            team_trades(team, season, G)

        elif choice == '3':
            most_connected(G)

        elif choice == '4':
            team = input("Enter team abbreviation (e.g. ATL): ").strip().upper()
            highest_received(team, G)

        elif choice == '5':
            print("Goodbye!")
            sys.exit()

        else:
            print("↳ Invalid choice; please enter 1–5.\n")
            continue

        print("\n" + "-" * 40 + "\n")

if __name__ == '__main__':
    main()