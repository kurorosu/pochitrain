# Changelog Layout

このディレクトリは最新でなくなった changelog を保持します.
`CHANGELOG.md` は最新の changelog を保持し, 旧履歴を本ディレクトリへ移動します.

## Files

- `1.5.x.md`: `v1.5.0` から `v1.5.9`.
- `1.4.x.md`: `v1.4.0` から `v1.4.9`.
- `1.3.x.md`: `v1.3.0` から `v1.3.9`.
- `1.2.x.md`: `v1.2.0` から `v1.2.9`.
- `1.1.x.md`: `v1.1.0` から `v1.1.9`.
- `1.0.x.md`: `v1.0.0` から `v1.0.9`.

## Update Flow

1. 新機能, 修正, リファクタは `CHANGELOG.md` の `[Unreleased]` に追記する.
2. PR本文を `tmp_PR.md` に作成するタイミングで, `[Unreleased]` も同時に更新する.
3. リリース時は `CHANGELOG.md` に新しいバージョン節を追加する.
4. 最新ではなくなった履歴を `changelogs/<major>.<minor>.x.md` へ移動する.
5. 各バージョンは `Added`, `Changed`, `Fixed`, `Removed` の順で記載する.
6. `CHANGELOG.md` の `Archived Changelogs` を更新する.
7. PR参照は `([#123](https://github.com/kurorosu/pochitrain/pull/123))` の明示リンク形式で記載する.
8. PR参照には **PR 番号** を使う (Issue 番号ではない). PR 作成前は `N/A.` とする.

## Notes

- `[Unreleased]` の未確定項目は `N/A.` を使う.
- リリース済みバージョンで該当項目がない場合は `なし.` を使う.
- 1行要約より, 利用者が判断できる粒度の変更内容を優先する.
