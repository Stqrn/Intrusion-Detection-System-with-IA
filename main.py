import os

def run_command(command):
    print(f"\n➡️ Running: {command}")
    os.system(command)

def main():
    while True:
        print("\n📊 Intrusion Detection System - Main Menu")
        print("1. 🔧 Train Random Forest model")
        print("2. 🔍 Interactive Prediction (Manual/CSV)")
        print("3. 📈 Show Evaluation Visuals")
        print("4. 🤖 Compare Multiple Models")
        print("5. 🛡️ Run Smart Firewall")
        print("6. 📊 Launch Dashboard")
        print("7. 🚪 Exit")

        choice = input("Enter your choice (1-7): ")

        if choice == "1":
            run_command("python main.py")
        elif choice == "2":
            run_command("python -m src.predict_interactive")
        elif choice == "3":
            print("📁 Check outputs/figures/ for graphs.")
        elif choice == "4":
            run_command("python train_multiple_models.py")
        elif choice == "5":
            run_command("sudo python -m src.smart_firewall")
        elif choice == "6":
            run_command("streamlit run dashboard.py")
        elif choice == "7":
            print("👋 Exiting. Goodbye!")
            break
        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()
