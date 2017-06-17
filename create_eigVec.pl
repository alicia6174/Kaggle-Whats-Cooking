#!/usr/bin/perl

while ($line = <STDIN>)
{
    if ($line =~ /value/)
    {
        last;
    }

    print $line;
}

