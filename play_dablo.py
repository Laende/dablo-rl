"""
Play Dablo interactively!

Simple launcher script for the interactive Dablo game.
"""

from dablo.ui import play_interactive_dablo


def main():
    """Main launcher function"""
    print("🎲 Welcome to Interactive Dablo!")
    print()

    # Simple menu
    print("Choose game mode:")
    print("1. Play vs NPC (Easy)")
    print("2. Play vs NPC (Medium)")
    print("3. Play vs NPC (Hard)")
    print("4. Play vs Human (2 players)")
    print()

    try:
        choice = input("Enter choice (1-4) or press Enter for default: ").strip()

        if choice == "1":
            print("🤖 Starting game vs Easy NPC...")
            play_interactive_dablo(vs_npc=True, difficulty="easy")
        elif choice == "2" or choice == "":
            print("🤖 Starting game vs Medium NPC...")
            play_interactive_dablo(vs_npc=True, difficulty="medium")
        elif choice == "3":
            print("🤖 Starting game vs Hard NPC...")
            play_interactive_dablo(vs_npc=True, difficulty="hard")
        elif choice == "4":
            print("👥 Starting 2-player game...")
            play_interactive_dablo(vs_npc=False)
        else:
            print("Invalid choice, starting default game...")
            play_interactive_dablo(vs_npc=True, difficulty="medium")

    except KeyboardInterrupt:
        print("\n👋 Thanks for playing!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")


if __name__ == "__main__":
    main()
