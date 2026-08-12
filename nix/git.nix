{ lib }:

let
  identities = {
    kryorains = {
      identity = "kryorains";
      name = "kryorains";
      email = "kryorains@kryorains.io";
      commitSigning = false;
    };
  };

  mkShellHook =
    {
      identity,
      name,
      email,
      commitSigning ? true,
    }:
    ''
      export GIT_IDENTITY=${lib.escapeShellArg identity}
      export GIT_AUTHOR_NAME=${lib.escapeShellArg name}
      export GIT_AUTHOR_EMAIL=${lib.escapeShellArg email}
      export GIT_COMMITTER_NAME=${lib.escapeShellArg name}
      export GIT_COMMITTER_EMAIL=${lib.escapeShellArg email}

      ${lib.optionalString (!commitSigning) ''
        export GIT_CONFIG_COUNT=1
        export GIT_CONFIG_KEY_0=commit.gpgSign
        export GIT_CONFIG_VALUE_0=false
      ''}

      echo "Git identity: $GIT_IDENTITY"
      echo "Author: $GIT_AUTHOR_NAME <$GIT_AUTHOR_EMAIL>"
    '';
in
{
  inherit identities mkShellHook;
}
