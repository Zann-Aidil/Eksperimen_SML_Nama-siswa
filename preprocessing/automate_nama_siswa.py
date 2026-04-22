from __future__ import annotations
import argparse, json, re
from pathlib import Path
import joblib, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def extract_title(name: str) -> str:
    if pd.isna(name): return "Unknown"
    m = re.search(r",\s*([^\.]+)\.", str(name)); title = m.group(1).strip() if m else "Unknown"
    title_map = {"Mlle":"Miss","Ms":"Miss","Mme":"Mrs","Lady":"Royalty","Countess":"Royalty","Dona":"Royalty","Sir":"Royalty","Don":"Royalty","Jonkheer":"Royalty","Capt":"Officer","Col":"Officer","Major":"Officer","Dr":"Officer","Rev":"Officer"}
    return title_map.get(title, title if title in {"Mr","Mrs","Miss","Master"} else "Rare")

def extract_ticket_prefix(ticket: str) -> str:
    if pd.isna(ticket): return "NONE"
    cleaned = re.sub(r"[\. /]", "", str(ticket)).upper()
    alpha = "".join(ch for ch in cleaned if ch.isalpha())
    return alpha if alpha else "NUMERIC"

def extract_cabin_deck(cabin: str) -> str:
    if pd.isna(cabin) or str(cabin).strip() == "": return "Unknown"
    return str(cabin).strip()[0].upper()

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Title"] = out["Name"].apply(extract_title)
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["TicketPrefix"] = out["Ticket"].apply(extract_ticket_prefix)
    out["CabinDeck"] = out["Cabin"].apply(extract_cabin_deck)
    out["Age*Class"] = out["Age"].fillna(out["Age"].median()) * out["Pclass"]
    out["FarePerPerson"] = out["Fare"].fillna(out["Fare"].median()) / out["FamilySize"].replace(0, 1)
    return out

def preprocess(train_path, test_path, output_dir):
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    train_fe, test_fe = add_features(train_df), add_features(test_df)
    feature_cols = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Title","FamilySize","IsAlone","TicketPrefix","CabinDeck","Age*Class","FarePerPerson"]
    num = ["Pclass","Age","SibSp","Parch","Fare","FamilySize","IsAlone","Age*Class","FarePerPerson"]
    cat = ["Sex","Embarked","Title","TicketPrefix","CabinDeck"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())]), num),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    X_train, y_train = train_fe[feature_cols], train_fe["Survived"].astype(int)
    X_test = test_fe[feature_cols]
    X_train_proc, X_test_proc = pre.fit_transform(X_train), pre.transform(X_test)
    feature_names = pre.get_feature_names_out().tolist()
    train_processed = pd.DataFrame(X_train_proc, columns=feature_names)
    train_processed.insert(0, "Survived", y_train.values)
    train_processed.insert(0, "PassengerId", train_fe["PassengerId"].values)
    test_processed = pd.DataFrame(X_test_proc, columns=feature_names)
    test_processed.insert(0, "PassengerId", test_fe["PassengerId"].values)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    train_processed.to_csv(output_dir / "train_processed.csv", index=False)
    test_processed.to_csv(output_dir / "test_processed.csv", index=False)
    joblib.dump(pre, output_dir / "preprocessor.joblib")
    meta = {"n_train_rows":int(train_df.shape[0]),"n_test_rows":int(test_df.shape[0]),"n_features_after_preprocessing":int(len(feature_names)),"target_column":"Survived","id_column":"PassengerId","raw_feature_columns":feature_cols,"transformed_feature_columns":feature_names}
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True); p.add_argument("--test", required=True); p.add_argument("--output-dir", required=True)
    a = p.parse_args(); print(json.dumps(preprocess(a.train, a.test, a.output_dir), indent=2))
if __name__ == "__main__":
    main()

print("Preprocessing selesai!")
print("Output disimpan di:", output_dir)
