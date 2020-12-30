#!/usr/bin/env python3

# Copyright 2020 David Robillard <d@drobilla.net>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THIS SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""
Write Sphinx markup from Doxygen XML.

Takes a path to a directory of XML generated by Doxygen, and emits a directory
with a reStructuredText file for every documented symbol.
"""

import argparse
import os
import sys
import textwrap
import xml.etree.ElementTree

__author__ = "David Robillard"
__date__ = "2020-11-18"
__email__ = "d@drobilla.net"
__license__ = "ISC"
__version__ = __date__.replace("-", ".")


def load_index(index_path):
    """
    Load the index from XML.

    :returns: A dictionary from ID to skeleton records with basic information
    for every documented entity.  Some records have an ``xml_filename`` key
    with the filename of a definition file.  These files will be loaded later
    to flesh out the records in the index.
    """

    root = xml.etree.ElementTree.parse(index_path).getroot()
    index = {}

    for compound in root:
        compound_id = compound.get("refid")
        compound_kind = compound.get("kind")
        compound_name = compound.find("name").text
        if compound_kind in ["dir", "file", "page"]:
            continue

        # Add record for compound (compounds appear only once in the index)
        assert compound_id not in index
        index[compound_id] = {
            "kind": compound_kind,
            "name": compound_name,
            "xml_filename": compound_id + ".xml",
            "children": [],
        }

        name_prefix = (
            ("%s::" % compound_name) if compound_kind == "namespace" else ""
        )

        for child in compound.findall("member"):
            if child.get("refid") in index:
                assert compound_kind == "group"
                continue

            # Everything has a kind and a name
            child_record = {
                "kind": child.get("kind"),
                "name": name_prefix + child.find("name").text,
            }

            if child.get("kind") == "enum":
                # Enums are not compounds, but we want to resolve the parent of
                # their values so they are not written as top level documents
                child_record["children"] = []

            if child.get("kind") == "enumvalue":
                # Remove namespace prefix
                child_record["name"] = child.find("name").text

            index[child.get("refid")] = child_record

    return index


def resolve_index(index, root):
    """
    Walk a definition document and extend the index for linking.

    This does two things: sets the "parent" and "children" fields of all
    applicable records, and sets the "strong" field of enums so that the
    correct Sphinx role can be used when referring to them.
    """

    def add_child(index, parent_id, child_id):
        parent = index[parent_id]
        child = index[child_id]

        if child["kind"] == "enumvalue":
            assert parent["kind"] == "enum"
            assert "parent" not in child or child["parent"] == parent_id
            child["parent"] = parent_id

        else:
            if parent["kind"] in ["class", "struct", "union"]:
                assert "parent" not in child or child["parent"] == parent_id
                child["parent"] = parent_id

        if child_id not in parent["children"]:
            parent["children"] += [child_id]

    compound = root.find("compounddef")
    compound_kind = compound.get("kind")

    if compound_kind == "group":
        for subgroup in compound.findall("innergroup"):
            add_child(index, compound.get("id"), subgroup.get("refid"))

        for klass in compound.findall("innerclass"):
            add_child(index, compound.get("id"), klass.get("refid"))

    for section in compound.findall("sectiondef"):
        if section.get("kind").startswith("private"):
            for member in section.findall("memberdef"):
                if member.get("id") in index:
                    del index[member.get("id")]
        else:
            for member in section.findall("memberdef"):
                member_id = member.get("id")
                add_child(index, compound.get("id"), member_id)

                if member.get("kind") == "enum":
                    index[member_id]["strong"] = member.get("strong") == "yes"
                    for value in member.findall("enumvalue"):
                        add_child(index, member_id, value.get("id"))


def sphinx_role(record, lang):
    """
    Return the Sphinx role used for a record.

    This is used for the description directive like ".. c:function::", and
    links like ":c:func:`foo`.
    """

    kind = record["kind"]

    if kind in ["class", "function", "namespace", "struct", "union"]:
        return lang + ":" + kind

    if kind == "define":
        return "c:macro"

    if kind == "enum":
        return lang + (":enum-class" if record["strong"] else ":enum")

    if kind == "typedef":
        return lang + ":type"

    if kind == "enumvalue":
        return lang + ":enumerator"

    if kind == "variable":
        return lang + (":member" if "parent" in record else ":var")

    raise RuntimeError("No known role for kind '%s'" % kind)


def child_identifier(lang, parent_name, child_name):
    """
    Return the identifier for an enum value or struct member.

    Sphinx, for some reason, uses a different syntax for this in C and C++.
    """

    separator = "::" if lang == "cpp" else "."

    return "%s%s%s" % (parent_name, separator, child_name)


def link_markup(index, lang, refid):
    """Return a Sphinx link for a Doxygen reference."""

    record = index[refid]
    kind, name = record["kind"], record["name"]
    role = sphinx_role(record, lang)

    if kind in ["class", "enum", "struct", "typedef", "union"]:
        return ":%s:`%s`" % (role, name)

    if kind == "function":
        return ":%s:func:`%s`" % (lang, name)

    if kind == "enumvalue":
        parent_name = index[record["parent"]]["name"]
        return ":%s:`%s`" % (role, child_identifier(lang, parent_name, name))

    if kind == "variable":
        if "parent" not in record:
            return ":%s:var:`%s`" % (lang, name)

        parent_name = index[record["parent"]]["name"]
        return ":%s:`%s`" % (role, child_identifier(lang, parent_name, name))

    raise RuntimeError("Unknown link target kind: %s" % kind)


def indent(markup, depth):
    """
    Indent markup to a depth level.

    Like textwrap.indent() but takes an integer and works in reST indentation
    levels for clarity."
    """

    return textwrap.indent(markup, "   " * depth)


def heading(text, level):
    """
    Return a ReST heading at a given level.

    Follows the style in the Python documentation guide, see
    <https://devguide.python.org/documenting/#sections>.
    """

    assert 1 <= level <= 6

    chars = ("#", "*", "=", "-", "^", '"')
    line = chars[level] * len(text)

    return "%s\n%s\n%s\n\n" % (line if level < 3 else "", text, line)


def dox_to_rst(index, lang, node):
    """
    Convert documentation commands (docCmdGroup) to Sphinx markup.

    This is used to convert the content of descriptions in the documentation.
    It recursively parses all children tags and raises a RuntimeError if any
    unknown tag is encountered.
    """

    def field_value(markup):
        """Return a value for a field as a single line or indented block."""
        if "\n" in markup.strip():
            return "\n" + indent(markup, 1)

        return " " + markup.strip()

    if node.tag == "computeroutput":
        assert len(node) == 0
        return "``%s``" % node.text

    if node.tag == "itemizedlist":
        markup = ""
        for item in node.findall("listitem"):
            assert len(item) == 1
            markup += "\n- %s" % dox_to_rst(index, lang, item[0])

        return markup

    if node.tag == "para":
        markup = node.text if node.text is not None else ""
        for child in node:
            markup += dox_to_rst(index, lang, child)
            markup += child.tail if child.tail is not None else ""

        return markup.strip() + "\n\n"

    if node.tag == "parameterlist":
        markup = ""
        for item in node.findall("parameteritem"):
            name = item.find("parameternamelist/parametername")
            description = item.find("parameterdescription")
            assert len(description) == 1
            markup += "\n\n:param %s:%s" % (
                name.text,
                field_value(dox_to_rst(index, lang, description[0])),
            )

        return markup + "\n"

    if node.tag == "programlisting":
        return "\n.. code-block:: %s\n\n%s" % (
            lang,
            indent(plain_text(node), 1),
        )

    if node.tag == "ref":
        refid = node.get("refid")
        if refid not in index:
            sys.stderr.write("warning: Unresolved link: %s\n" % refid)
            return node.text

        assert len(node) == 0
        assert len(link_markup(index, lang, refid)) > 0
        return link_markup(index, lang, refid)

    if node.tag == "simplesect":
        assert len(node) == 1

        if node.get("kind") == "return":
            return "\n:returns:" + field_value(
                dox_to_rst(index, lang, node[0])
            )

        if node.get("kind") == "see":
            return dox_to_rst(index, lang, node[0])

        raise RuntimeError("Unknown simplesect kind: %s" % node.get("kind"))

    if node.tag == "ulink":
        return "`%s <%s>`_" % (node.text, node.get("url"))

    raise RuntimeError("Unknown documentation command: %s" % node.tag)


def description_markup(index, lang, node):
    """Return the markup for a brief or detailed description."""

    assert node.tag == "briefdescription" or node.tag == "detaileddescription"
    assert not (node.tag == "briefdescription" and len(node) > 1)
    assert len(node.text.strip()) == 0

    return "".join([dox_to_rst(index, lang, child) for child in node])


def set_descriptions(index, lang, definition, record):
    """Set a record's brief/detailed descriptions from the XML definition."""

    for tag in ["briefdescription", "detaileddescription"]:
        node = definition.find(tag)
        if node is not None:
            record[tag] = description_markup(index, lang, node)


def set_template_params(node, record):
    """Set a record's template_params from the XML definition."""

    template_param_list = node.find("templateparamlist")
    if template_param_list is not None:
        params = []
        for param in template_param_list.findall("param"):
            if param.find("declname") is not None:
                # Value parameter
                type_text = plain_text(param.find("type"))
                name_text = plain_text(param.find("declname"))

                params += ["%s %s" % (type_text, name_text)]
            else:
                # Type parameter
                params += ["%s" % (plain_text(param.find("type")))]

        record["template_params"] = "%s" % ", ".join(params)


def plain_text(node):
    """
    Return the plain text of a node with all tags ignored.

    This is needed where Doxygen may include refs but Sphinx needs plain text
    because it parses things itself to generate links.
    """

    if node.tag == "sp":
        markup = " "
    elif node.text is not None:
        markup = node.text
    else:
        markup = ""

    for child in node:
        markup += plain_text(child)
        markup += child.tail if child.tail is not None else ""

    return markup


def local_name(name):
    """Return a name with all namespace prefixes stripped."""

    return name[name.rindex("::") + 2 :] if "::" in name else name


def read_definition_doc(index, lang, root):
    """Walk a definition document and update described records in the index."""

    # Set descriptions for the compound itself
    compound = root.find("compounddef")
    compound_record = index[compound.get("id")]
    set_descriptions(index, lang, compound, compound_record)
    set_template_params(compound, compound_record)

    if compound.find("title") is not None:
        compound_record["title"] = compound.find("title").text.strip()

    # Set documentation for all children
    for section in compound.findall("sectiondef"):
        if section.get("kind").startswith("private"):
            continue

        for member in section.findall("memberdef"):
            kind = member.get("kind")
            record = index[member.get("id")]
            set_descriptions(index, lang, member, record)
            set_template_params(member, record)

            if compound.get("kind") in ["class", "struct", "union"]:
                assert kind in ["function", "typedef", "variable"]
                record["type"] = plain_text(member.find("type"))

            if kind == "enum":
                for value in member.findall("enumvalue"):
                    set_descriptions(
                        index, lang, value, index[value.get("id")]
                    )

            elif kind == "function":
                record["prototype"] = "%s %s%s" % (
                    plain_text(member.find("type")),
                    member.find("name").text,
                    member.find("argsstring").text,
                )

            elif kind == "typedef":
                name = local_name(record["name"])
                args_text = member.find("argsstring").text
                target_text = plain_text(member.find("type"))
                if args_text is not None:  # Function pointer
                    assert target_text[-2:] == "(*" and args_text[0] == ")"
                    record["type"] = target_text + args_text
                    record["definition"] = target_text + name + args_text
                else:  # Normal named typedef
                    assert target_text is not None
                    record["type"] = target_text
                    if member.find("definition").text.startswith("using"):
                        record["definition"] = "%s = %s" % (
                            name,
                            target_text,
                        )
                    else:
                        record["definition"] = "%s %s" % (
                            target_text,
                            name,
                        )


def declaration_string(record):
    """
    Return the string that describes a declaration.

    This is what follows the directive, and is in C/C++ syntax, except without
    keywords like "typedef" and "using" as expected by Sphinx.  For example,
    "struct ThingImpl Thing" or "void run(int value)".
    """

    kind = record["kind"]
    result = ""

    if "template_params" in record:
        result = "template <%s> " % record["template_params"]

    if kind == "function":
        result += record["prototype"]
    elif kind == "typedef":
        result += record["definition"]
    elif "type" in record:
        result += "%s %s" % (record["type"], local_name(record["name"]))
    else:
        result += local_name(record["name"])

    return result


def document_markup(index, lang, record):
    """Return the complete document that describes some documented entity."""

    kind = record["kind"]
    role = sphinx_role(record, lang)
    name = record["name"]
    markup = ""

    if name != local_name(name):
        markup += ".. cpp:namespace:: %s\n\n" % name[0 : name.rindex("::")]

    # Write top-level directive
    markup += ".. %s:: %s\n" % (role, declaration_string(record))

    # Write main description blurb
    markup += "\n"
    markup += indent(record["briefdescription"], 1)
    markup += indent(record["detaileddescription"], 1)

    assert (
        kind in ["class", "enum", "namespace", "struct", "union"]
        or "children" not in record
    )

    # Sphinx C++ namespaces work by setting a scope, they have no content
    child_indent = 0 if kind == "namespace" else 1

    # Write inline children if applicable
    markup += "\n"
    for child_id in record.get("children", []):
        child_record = index[child_id]
        child_role = sphinx_role(child_record, lang)

        child_header = ".. %s:: %s\n\n" % (
            child_role,
            declaration_string(child_record),
        )

        markup += "\n"
        markup += indent(child_header, child_indent)
        markup += indent(child_record["briefdescription"], child_indent + 1)
        markup += indent(child_record["detaileddescription"], child_indent + 1)
        markup += "\n"

    return markup


def symbol_filename(name):
    """Adapt the name of a symbol to be suitable for use as a filename."""

    return name.replace("::", "__")


def emit_symbols(index, lang, symbol_dir, force):
    """Write a description file for every symbol documented in the index."""

    for record in index.values():
        if (
            record["kind"] in ["group", "namespace"]
            or "parent" in record
            and index[record["parent"]]["kind"] != "group"
        ):
            continue

        name = record["name"]
        filename = os.path.join(symbol_dir, symbol_filename("%s.rst" % name))
        if not force and os.path.exists(filename):
            raise FileExistsError("File already exists: '%s'" % filename)

        with open(filename, "w") as rst:
            rst.write(heading(local_name(name), 3))
            rst.write(document_markup(index, lang, record))


def emit_groups(index, output_dir, symbol_dir_name, force):
    """Write a description file for every group documented in the index."""

    for record in index.values():
        if record["kind"] != "group":
            continue

        name = record["name"]
        filename = os.path.join(output_dir, "%s.rst" % name)
        if not force and os.path.exists(filename):
            raise FileExistsError("File already exists: '%s'" % filename)

        with open(filename, "w") as rst:
            rst.write(heading(record["title"], 2))

            # Get all child group and symbol names
            group_names = []
            symbol_names = []
            for child_id in record["children"]:
                child = index[child_id]
                if child["kind"] == "group":
                    group_names += [child["name"]]
                else:
                    symbol_names += [child["name"]]

            # Emit description (document body)
            rst.write(record["briefdescription"] + "\n\n")
            rst.write(record["detaileddescription"] + "\n\n")

            # Emit TOC
            rst.write(".. toctree::\n")

            # Emit groups at the top of the TOC
            for group_name in group_names:
                rst.write("\n" + indent(group_name, 1))

            # Emit symbols in sorted order
            for symbol_name in sorted(symbol_names):
                path = "/".join(
                    [symbol_dir_name, symbol_filename(symbol_name)]
                )
                rst.write("\n" + indent(path, 1))

            rst.write("\n")


def run(index_xml_path, output_dir, symbol_dir_name, language, force):
    """Write a directory of Sphinx files from a Doxygen XML directory."""

    # Build skeleton index from index.xml
    xml_dir = os.path.dirname(index_xml_path)
    index = load_index(index_xml_path)

    # Load all definition documents
    definition_docs = []
    for record in index.values():
        if "xml_filename" in record:
            xml_path = os.path.join(xml_dir, record["xml_filename"])
            definition_docs += [xml.etree.ElementTree.parse(xml_path)]

    # Do an initial pass of the definition documents to resolve the index
    for root in definition_docs:
        resolve_index(index, root)

    # Finally read the documentation from definition documents
    for root in definition_docs:
        read_definition_doc(index, language, root)

    # Emit output files
    symbol_dir = os.path.join(output_dir, symbol_dir_name)
    os.makedirs(symbol_dir, exist_ok=True)
    emit_symbols(index, language, symbol_dir, force)
    emit_groups(index, output_dir, symbol_dir_name, force)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        usage="%(prog)s [OPTION]... XML_DIR OUTPUT_DIR",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite files",
    )

    ap.add_argument(
        "-l",
        "--language",
        default="c",
        choices=["c", "cpp"],
        help="language domain for output",
    )

    ap.add_argument(
        "-s",
        "--symbol-dir-name",
        default="symbols",
        help="name for subdirectory of symbol documentation files",
    )

    ap.add_argument("index_xml_path", help="path index.xml from Doxygen")
    ap.add_argument("output_dir", help="output directory")

    run(**vars(ap.parse_args(sys.argv[1:])))
