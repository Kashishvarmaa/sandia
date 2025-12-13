"""
Shell Script AST Parser
Converts shell scripts to Abstract Syntax Trees using bashlex
"""

import re
import bashlex
from typing import List, Dict, Any, Optional


class ShellScriptParser:
    """Parse shell scripts into AST"""

    def __init__(self):
        # Command risk levels (manually curated)
        self.command_risk = {
            # High risk (network/execution)
            'curl': 0.8, 'wget': 0.8, 'nc': 0.9, 'netcat': 0.9,
            'bash': 0.7, 'sh': 0.7, 'eval': 0.9, 'exec': 0.8,
            'nohup': 0.7, 'setsid': 0.6,

            # Medium-high risk (file operations)
            'chmod': 0.6, 'chown': 0.5, 'rm': 0.6, 'dd': 0.7,
            'mv': 0.4, 'cp': 0.3,

            # Medium risk (system modification)
            'systemctl': 0.5, 'service': 0.5, 'crontab': 0.7,
            'useradd': 0.6, 'usermod': 0.6, 'passwd': 0.7,

            # Low-medium risk (info gathering)
            'ps': 0.3, 'top': 0.2, 'netstat': 0.3, 'ifconfig': 0.3,
            'whoami': 0.3, 'id': 0.3, 'uname': 0.3,

            # Low risk (common utilities)
            'echo': 0.1, 'cat': 0.2, 'grep': 0.2, 'awk': 0.2,
            'sed': 0.2, 'ls': 0.1, 'cd': 0.1, 'pwd': 0.1,
        }

    def parse(self, script_content: str) -> Optional[List[Dict[str, Any]]]:
        """
        Parse shell script to AST

        Args:
            script_content: Shell script as string

        Returns:
            List of AST nodes or None if parsing fails
        """
        try:
            # Remove shebang and comments for cleaner parsing
            cleaned = self._clean_script(script_content)

            # Parse with bashlex
            nodes = bashlex.parse(cleaned)

            # Convert to simpler representation
            simplified = []
            for node in nodes:
                simplified.extend(self._traverse_node(node))

            return simplified

        except Exception as e:
            print(f"[AST Parser] bashlex failed: {e}")
            # Fallback to regex-based parsing
            return self._fallback_parse(script_content)

    def _clean_script(self, script: str) -> str:
        """Remove comments and clean script"""
        lines = []
        for line in script.split('\n'):
            # Skip shebang
            if line.startswith('#!'):
                continue
            # Remove inline comments (preserve strings)
            line = re.sub(r'(?<!\\)#.*$', '', line)
            lines.append(line)
        return '\n'.join(lines)

    def _traverse_node(self, node, depth=0) -> List[Dict[str, Any]]:
        """Recursively traverse AST node"""
        nodes = []

        node_type = node.kind

        if node_type == 'command':
            # Extract command info
            parts = []
            if hasattr(node, 'parts'):
                for part in node.parts:
                    if part.kind == 'word':
                        parts.append(part.word)

            if parts:
                command = parts[0]
                args = parts[1:] if len(parts) > 1 else []

                nodes.append({
                    'type': 'command',
                    'command': command,
                    'args': args,
                    'risk': self.command_risk.get(command, 0.3),
                    'depth': depth
                })

        elif node_type == 'pipeline':
            # Commands connected by pipes
            if hasattr(node, 'parts'):
                for part in node.parts:
                    nodes.extend(self._traverse_node(part, depth))

        elif node_type == 'compound':
            # Compound statements (if, while, for, etc.)
            if hasattr(node, 'list'):
                for item in node.list:
                    nodes.extend(self._traverse_node(item, depth + 1))

        elif node_type == 'if':
            # If statements
            nodes.append({'type': 'conditional', 'kind': 'if', 'depth': depth})
            if hasattr(node, 'parts'):
                for part in node.parts:
                    if hasattr(part, 'list'):
                        for item in part.list:
                            nodes.extend(self._traverse_node(item, depth + 1))

        elif node_type == 'while':
            # While loops
            nodes.append({'type': 'loop', 'kind': 'while', 'depth': depth})
            if hasattr(node, 'parts'):
                for part in node.parts:
                    if hasattr(part, 'list'):
                        for item in part.list:
                            nodes.extend(self._traverse_node(item, depth + 1))

        elif node_type == 'for':
            # For loops
            nodes.append({'type': 'loop', 'kind': 'for', 'depth': depth})
            if hasattr(node, 'parts'):
                for part in node.parts:
                    if hasattr(part, 'list'):
                        for item in part.list:
                            nodes.extend(self._traverse_node(item, depth + 1))

        # Handle other node types recursively
        if hasattr(node, 'parts') and node_type not in ['command', 'pipeline', 'if', 'while', 'for']:
            for part in node.parts:
                nodes.extend(self._traverse_node(part, depth))

        if hasattr(node, 'list') and node_type not in ['compound', 'if', 'while', 'for']:
            for item in node.list:
                nodes.extend(self._traverse_node(item, depth))

        return nodes

    def _fallback_parse(self, script_content: str) -> List[Dict[str, Any]]:
        """
        Regex-based fallback parser when bashlex fails
        Used for complex or non-standard shell syntax
        """
        nodes = []
        lines = script_content.split('\n')

        for idx, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Detect control structures
            if re.match(r'\bif\b', line):
                nodes.append({'type': 'conditional', 'kind': 'if', 'line': idx})
            elif re.match(r'\bwhile\b|\buntil\b', line):
                nodes.append({'type': 'loop', 'kind': 'while', 'line': idx})
            elif re.match(r'\bfor\b', line):
                nodes.append({'type': 'loop', 'kind': 'for', 'line': idx})

            # Detect commands
            command_match = re.match(r'([a-zA-Z0-9_\-\.\/]+)', line)
            if command_match:
                command = command_match.group(1).split('/')[-1]  # Get command name
                nodes.append({
                    'type': 'command',
                    'command': command,
                    'args': [],
                    'risk': self.command_risk.get(command, 0.3),
                    'line': idx
                })

        return nodes

    def get_command_sequence(self, ast_nodes: List[Dict]) -> List[str]:
        """Extract command sequence from AST"""
        commands = []
        for node in ast_nodes:
            if node.get('type') == 'command':
                commands.append(node['command'])
        return commands

    def get_control_flow_complexity(self, ast_nodes: List[Dict]) -> Dict[str, int]:
        """Calculate control flow complexity metrics"""
        metrics = {
            'total_commands': 0,
            'conditionals': 0,
            'loops': 0,
            'max_depth': 0,
            'avg_risk': 0.0
        }

        risks = []
        for node in ast_nodes:
            if node.get('type') == 'command':
                metrics['total_commands'] += 1
                risks.append(node.get('risk', 0.3))
            elif node.get('type') == 'conditional':
                metrics['conditionals'] += 1
            elif node.get('type') == 'loop':
                metrics['loops'] += 1

            # Track max depth
            if 'depth' in node:
                metrics['max_depth'] = max(metrics['max_depth'], node['depth'])

        if risks:
            metrics['avg_risk'] = sum(risks) / len(risks)

        return metrics
